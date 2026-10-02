"""Evolve the cell rule (cell.py) for a robot task (robot.py), by CMA-ES.

    python evolve.py TASK [GENERATIONS] [--from KERNEL] [--sparsity S] [--sigma SIGMA] [--sigma-m SIGMA_M]

Each generation, every candidate rule lives LIVES lives in fresh random worlds, at each tick size in DTS; the fitness
is the worse of its task scores over the tick sizes (so no rule can rely on the tick size), minus S x the sum of the
rule's absolute coefficients (so rules stay short enough to say). The rule of the search's mean is scored the same
way on fixed validation worlds each generation, also at the finer tick sizes in CHECK_DTS, and whenever it beats the
best so far it is saved to kernels/TASK.json.

Fitness per task (averaged over lives), less the cost of activity (`cost`): FIRE_COST per thruster firing, and
ACTIVITY x how far the sheet's spike rate (spikes per cell per unit time) is above SPARSE_RATE. Sparse activity is
free; a wave through every cell is not. In move and approach activity is free (ACTIVITY_FREE): with it charged, the
easiest improvement on a moving robot is to go quiet, not to approach:
  move      how far the robot ends from where it started, in robot radii
  approach  blocks eaten, plus how much nearer the one it is heading for it ended than it started (0..1)
  taste     for each world, lived once with its one (red) block as food and once as poison: the sum of the two
            lives' food - POISON x poison. With POISON below 1, eating everything scores above standing still (a
            stepping stone from approach); learning to stop after a bad taste scores more
  choose    for each world, lived once with red as food and once with blue: the worse of the two lives' score. A
            fixed colour preference scores below standing still (its wrong-colour life); learning scores best
  forage    as choose, with several blocks of each colour
  graze     several blocks of each colour, all food, seen only near by: blocks eaten (a robot that gets around
            among many blocks, before any learning is asked of it)
  discriminate  as graze, one colour food and the other poison, lived once each way and long enough for tens of
            tastes: the worse of the two lives' score over the last third of life only (what it ends up doing)

Lives are counted in thirds (robot.PARTS). For taste, choose, forage and discriminate, learning is also scored
directly: LEARN x (the food share of the world's meals in the last third of its lives - the share in the first),
when both have meals. Eating everything scores 0 on this, and a rule that learns even a little scores above it, so
the search can find its way off that plateau.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cell
import robot

DURATION = {"move": 400, "approach": 1000, "taste": 2000, "choose": 2000, "forage": 3000,   # simulated time per life
            "graze": 2000, "discriminate": 6000}
LEARNING_TASKS = ("taste", "choose", "forage", "discriminate")      # lived in pairs: each colour as food
LIVES = 16
POISON = 0.5
FIRE_COST = 0.001
ACTIVITY = 30.0
ACTIVITY_FREE = ("move", "approach")    # getting the robot going comes first: activity costs from graze on
SPARSE_RATE = 0.01
LEARN = 2.0
VALIDATION = 64
DTS = (1.0, 0.5, 0.25)          # tick sizes every candidate lives at (the score is the worst over them)
CHECK_DTS = (1.0, 0.5, 0.25, 0.125)     # ... and validation, finer still than any it was evolved at
POPULATION = 32
KERNELS = Path("kernels")

_geometry = None


def geometry() -> robot.Geometry:
    """The robot's body, built once per process."""
    global _geometry
    if _geometry is None:
        _geometry = robot.Geometry()
    return _geometry


def assess(rule: np.ndarray, task: str, seeds, dt: float = 1.0, world: robot.World = robot.World()) -> dict:
    """Live a rule in each world at tick size dt, all lives together: for the LEARNING_TASKS each world twice (red as
    food, then blue), otherwise once (food alternating; graze: both). Returns, per world, its value ("values": a
    life's task score less the cost of its activity; for a pair of lives, their sum for taste and the worse
    otherwise) and ("parts") the (food, poison) eaten in each part of its lives, (worlds, PARTS, 2); and the spike rate
    of each life ("spikes")."""
    world = dataclasses.replace(world, dt=dt)
    pairs = task in LEARNING_TASKS
    if pairs:
        worlds = [(s, f) for s in seeds for f in (robot.RED, robot.BLUE)]
    else:
        worlds = [(s, robot.BOTH if task == "graze" else s % 2) for s in seeds]
    r = robot.lifetimes(rule, geometry(), task, worlds, world, DURATION[task])
    values = []
    for i, counts in enumerate(r["counts"]):
        food_eaten, poison = counts[:, :2].sum(axis=0)
        if task == "move":
            value = float(np.hypot(*(r["end"][i] - r["start"][i]))) / world.radius
        elif task == "approach":
            # partial credit: how much nearer the current block the robot ended than the block started (a block that
            # reappeared after being eaten counts from a typical 12)
            distance = float(np.hypot(*(r["last"][i, 0, :2] - r["end"][i])))
            reference = float(np.hypot(*(r["first"][i, 0, :2] - r["start"][i]))) if food_eaten == 0 else 12.0
            value = float(food_eaten) + max(0.0, 1.0 - distance / reference)
        elif task == "discriminate":
            value = float(counts[-1, 0] - POISON * counts[-1, 1])
        else:
            value = float(food_eaten - POISON * poison)
        values.append(value - (0.0 if task in ACTIVITY_FREE else cost(counts, geometry().sheet.n, DURATION[task])))
    values = np.array(values)
    parts = r["counts"][:, :, :2]
    if pairs:
        values = values.reshape(-1, 2).sum(axis=1) if task == "taste" else values.reshape(-1, 2).min(axis=1)
        parts = parts.reshape(-1, 2, robot.PARTS, 2).sum(axis=1)
    spikes = np.array([spike_rate(c, geometry().sheet.n, DURATION[task]) for c in r["counts"]])
    return {"values": values, "parts": parts, "spikes": spikes}


def cost(counts, cells: int, duration: float) -> float:
    """The cost of a life's activity: its thruster firings, and its spike rate (per cell per unit of simulated time,
    so the same at any tick size) above SPARSE_RATE."""
    return FIRE_COST * counts[:, 2].sum() + ACTIVITY * max(0.0, spike_rate(counts, cells, duration) - SPARSE_RATE)


def spike_rate(counts, cells: int, duration: float) -> float:
    """Spikes per cell per unit of simulated time."""
    return counts[:, 3].sum() / (cells * duration)


def learned(parts, lives: int, minimum: float = 0.0) -> float:
    """The rise in food share from the first part of lives to the last; 0 unless both have more than minimum
    meals per life."""
    (f1, p1), (f2, p2) = parts[0], parts[-1]
    if min(f1 + p1, f2 + p2) <= minimum * lives:
        return 0.0
    return f2 / (f2 + p2) - f1 / (f1 + p1)


def score(rule: np.ndarray, task: str, seeds, dts=DTS) -> float:
    """The task's score: the worse over the tick sizes of the mean over the worlds."""
    return min(score_at(rule, task, seeds, dt) for dt in dts)


def score_at(rule: np.ndarray, task: str, seeds, dt: float) -> float:
    a = assess(rule, task, seeds, dt)
    total = a["values"].sum()
    if task in LEARNING_TASKS:
        total += LEARN * sum(learned(parts, 2) for parts in a["parts"])
    return float(total / len(seeds))


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
            jobs = [(mean, task, s, dt) for dt in CHECK_DTS for s in validation]
            per = np.array(pool.map(_score_one, jobs)).reshape(len(CHECK_DTS), len(validation)).mean(axis=1)
            valid = float(per.min())
            learning = ""
            if task in LEARNING_TASKS:
                eaten = np.sum(pool.map(_parts, [(mean, task, s) for s in validation]), axis=0)
                share = lambda f, p: f / max(1, f + p)
                learning = (f"  food share: first third {share(*eaten[0]):.0%} of {eaten[0].sum()}, "
                            f"last third {share(*eaten[-1]):.0%} of {eaten[-1].sum()}")
            if valid > best:
                best = valid
                (KERNELS / f"{task}.json").write_text(json.dumps(
                    {"task": task, "generation": generation, "validation": valid,
                     "validation_by_dt": dict(zip(map(str, CHECK_DTS), per.tolist())), "variables": cell.VARIABLES,
                     "rule": mean.tolist()}, indent=1))
            print(f"gen {generation:4d}  fitness best {max(fitness):7.3f} mean {np.mean(fitness):7.3f}  "
                  f"validation {valid:7.3f} (by dt {' '.join(f'{v:.2f}' for v in per)}; best {best:.3f})  sigma {es.sigma:.3f}  |rule| {np.abs(mean).sum():.2f}{learning}",
                  flush=True)


def _parts(args):
    """(food, poison) in each part of life, summed over a world's red-food and blue-food lives."""
    rule, task, seed = args
    return assess(rule, task, [seed])["parts"][0]


def _score_one(args):
    rule, task, seed, dt = args
    return score_at(rule, task, [seed], dt)


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
