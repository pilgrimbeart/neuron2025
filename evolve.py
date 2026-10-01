"""Evolve the cell rule (cell.py) for a robot task (robot.py), by CMA-ES.

    python evolve.py TASK [GENERATIONS] [--from KERNEL] [--sparsity S] [--sigma SIGMA] [--sigma-m SIGMA_M]

Each generation, every candidate rule lives LIVES lives in fresh random worlds; the fitness is the task's score minus
S x the sum of the rule's absolute coefficients (so rules stay short enough to say). The rule of the search's mean
is scored on fixed validation worlds each generation, and whenever it beats the best so far it is saved to
kernels/TASK.json.

Fitness per task (averaged over lives), less FIRE_COST per thruster firing (so quiet, sparse activity is favoured):
  move      how far the robot ends from where it started, in robot radii
  approach  blocks eaten, plus how much nearer the one it is heading for it ended than it started (0..1)
  taste     for each world, lived once with its one (red) block as food and once as poison: the sum of the two
            lives' food - POISON x poison. With POISON below 1, eating everything scores above standing still (a
            stepping stone from approach); learning to stop after a bad taste scores more
  choose    for each world, lived once with red as food and once with blue: the worse of the two lives' score. A
            fixed colour preference scores below standing still (its wrong-colour life); learning scores best
  forage    as choose, with several blocks of each colour

For taste, choose and forage, learning is also scored directly: LEARN x (the food share of the world's meals in the
second half of its lives - the share in the first half), when both halves have meals. Eating everything scores 0 on
this, and a rule that learns even a little scores above it, so the search can find its way off that plateau.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cell
import robot

DURATION = {"move": 400, "approach": 1000, "taste": 2000, "choose": 2000, "forage": 3000}   # simulated time per life
LIVES = 16
POISON = 0.5
FIRE_COST = 0.001
LEARN = 2.0
VALIDATION = 64
POPULATION = 32
KERNELS = Path("kernels")

_geometry = None


def life(rule: np.ndarray, task: str, seed: int, food: int, halves: bool = False, world: robot.World = robot.World()):
    """The task's score for one life, less the firing cost; with halves, also the (food, poison) eaten in each half."""
    global _geometry
    if _geometry is None:
        _geometry = robot.Geometry()
    per_half, start, pos, first, blocks = robot.lifetime(rule, _geometry, task, seed, food, world, DURATION[task])
    food_eaten, poison, firings = per_half.sum(axis=0)
    if task == "move":
        value = float(np.hypot(*(pos - start))) / world.radius
    elif task == "approach":
        # partial credit: how much nearer the current block the robot ended than the block started (a block that
        # reappeared after being eaten counts from a typical 12)
        distance = float(np.hypot(*(blocks[0, :2] - pos)))
        reference = float(np.hypot(*(first[0, :2] - start))) if food_eaten == 0 else 12.0
        value = float(food_eaten) + max(0.0, 1.0 - distance / reference)
    else:
        value = float(food_eaten - POISON * poison)
    value -= FIRE_COST * firings
    return (value, per_half[:, :2]) if halves else value


def score(rule: np.ndarray, task: str, seeds) -> float:
    if task in ("move", "approach"):
        return float(np.mean([life(rule, task, s, s % 2) for s in seeds]))
    total = 0.0
    for s in seeds:
        (a, halves_a), (b, halves_b) = (life(rule, task, s, food, halves=True) for food in (robot.RED, robot.BLUE))
        total += a + b if task == "taste" else min(a, b)
        (f1, p1), (f2, p2) = halves_a + halves_b
        if f1 + p1 and f2 + p2:
            total += LEARN * (f2 / (f2 + p2) - f1 / (f1 + p1))
    return total / len(seeds)


def _job(args):
    rule, task, seeds, sparsity = args
    return score(rule, task, seeds) - sparsity * float(np.abs(rule).sum())


def involves_m() -> np.ndarray:
    """Which coefficients have to do with m: m's own, and every term with m or nm in it."""
    mask = np.zeros(cell.N_PARAMS, dtype=bool)
    for a, name in enumerate(cell.VARIABLES):
        for c, term in enumerate(cell.TERMS):
            mask[a * cell.N_TERMS + c] = name == "m" or "m" in term
    return mask


def evolve(task: str, generations: int, start: np.ndarray | None, sparsity: float, sigma: float = 1.0,
           sigma_m: float | None = None) -> None:
    """sigma_m, if given, is the starting spread for the coefficients to do with m (so a rule found without m can be
    searched gently while m's new terms explore)."""
    import cma
    KERNELS.mkdir(exist_ok=True)
    x0 = np.zeros(cell.N_PARAMS) if start is None else start
    options = {"popsize": POPULATION, "verbose": -9, "seed": 1}
    if sigma_m is not None:
        options["CMA_stds"] = np.where(involves_m(), sigma_m / sigma, 1.0)
    es = cma.CMAEvolutionStrategy(x0, sigma, options)
    validation = range(10**6, 10**6 + VALIDATION)
    best = -math.inf
    rng = np.random.default_rng(0)
    with Pool() as pool:
        for generation in range(generations):
            candidates = es.ask()
            seeds = [int(s) for s in rng.integers(0, 10**6, LIVES)]
            fitness = pool.map(_job, [(np.asarray(c), task, seeds, sparsity) for c in candidates])
            es.tell(candidates, [-f for f in fitness])
            mean = np.asarray(es.mean)
            valid = float(np.mean(pool.map(_score_one, [(mean, task, [s]) for s in validation])))
            learning = ""
            if task in ("taste", "choose", "forage"):
                eaten = np.sum(pool.map(_halves, [(mean, task, s) for s in validation]), axis=0)
                share = lambda f, p: f / max(1, f + p)
                learning = (f"  food share: first half {share(*eaten[0]):.0%} of {eaten[0].sum()}, "
                            f"second half {share(*eaten[1]):.0%} of {eaten[1].sum()}")
            if valid > best:
                best = valid
                (KERNELS / f"{task}.json").write_text(json.dumps(
                    {"task": task, "generation": generation, "validation": valid, "variables": cell.VARIABLES,
                     "rule": mean.tolist()}, indent=1))
            print(f"gen {generation:4d}  fitness best {max(fitness):7.3f} mean {np.mean(fitness):7.3f}  "
                  f"validation {valid:7.3f} (best {best:.3f})  sigma {es.sigma:.3f}  |rule| {np.abs(mean).sum():.2f}{learning}",
                  flush=True)


def _halves(args):
    """(food, poison) in each half, summed over a world's red-food and blue-food lives."""
    rule, task, seed = args
    return sum(life(rule, task, seed, food, halves=True)[1] for food in (robot.RED, robot.BLUE))


def _score_one(args):
    rule, task, seeds = args
    return score(rule, task, seeds)


def load(path) -> np.ndarray:
    """A saved rule, widened to the current variables if it was evolved with fewer (the new terms zero)."""
    saved = json.loads(Path(path).read_text())
    return cell.widen(np.array(saved["rule"]), tuple(saved.get("variables", ("v", "w"))))


def load_world(path) -> robot.World:
    """The world a saved rule was evolved in: the default, with any body genes it carries."""
    saved = json.loads(Path(path).read_text())
    return robot.World(**saved.get("body", {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=robot.TASKS)
    parser.add_argument("generations", type=int, nargs="?", default=100)
    parser.add_argument("--from", dest="start")
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=1.0, help="the search's starting spread (smaller from a kernel)")
    parser.add_argument("--sigma-m", type=float, help="the starting spread for the coefficients to do with m")
    args = parser.parse_args()
    evolve(args.task, args.generations, load(args.start) if args.start else None, args.sparsity, args.sigma,
           args.sigma_m)
    print(cell.describe(load(KERNELS / f"{args.task}.json")))
