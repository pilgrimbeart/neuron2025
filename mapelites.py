"""MAP-Elites: a wide search for a cell rule that learns, keeping the best rule for each kind of behaviour rather
than only the best rule.

    python mapelites.py [HOURS] [--task taste|choose] [--resume]

A genome is a rule (cell.py) and two body genes: how strongly a taste kicks the cell where the block touched and
the middle of the disc (robot.World.taste_contact, taste_centre; 0..TASTE_MAX). Each genome lives in WORLDS fixed
worlds, each once with red as food and once with blue (in taste, the one red block is food, then poison), so every
evaluation is deterministic and an elite can't hold its place by luck.

Behaviour, 3 axes of BINS bins each:
  meals     meals per life, 0..MAX_MEALS
  food      how eating food changes from the first half of a life to the second, (F2 - F1) / (F1 + F2 + 2)
  poison    the same for poison, (P2 - P1) / (P1 + P2 + 2)
A learner sits where poison falls and food doesn't; eating everything, in the middle. Within a cell the higher score
wins, scored as in evolve.py, except that learning (the rise in food share from the first half of lives to the
second) only counts when both halves have at least MIN_MEALS meals per life, so that stopping eating can't pass for
learning. Offspring: a random elite, mutated at a random scale, or crossed with another along the line between them.

The archive is saved to kernels/map_TASK.pkl, the best-scoring elite to kernels/map_TASK.json and the best learner to
kernels/map_TASK_learner.json. Fixed worlds let flukes in, so every VALIDATE_EVERY batches the best learner is lived
again in VALIDATION fresh worlds; one that learns there too is saved to kernels/map_TASK_validated.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cell
import evolve
import robot

WORLDS = range(32)
VALIDATION = range(10**6, 10**6 + 64)
VALIDATE_EVERY = 50
MIN_MEALS = 0.25
BINS = 8
MAX_MEALS = 8.0
CHANGE = 0.6                # the food and poison axes span -CHANGE..CHANGE
TASTE_MAX = 1.5
BATCH = 64
INITIAL = 256
N_GENES = cell.N_PARAMS + 2

_geometry = None


def evaluate(args):
    """(score, behaviour (meals, food change, poison change), learning, meals per life, halves) for a genome, where
    halves is the (food, poison) eaten in the first and second halves of its lives."""
    genome, task, worlds = args
    global _geometry
    if _geometry is None:
        _geometry = robot.Geometry()
    rule = genome[:cell.N_PARAMS]
    world = robot.World(taste_contact=float(genome[-2]), taste_centre=float(genome[-1]))
    halves = np.zeros((2, 2))                   # (first, second half) x (food, poison)
    firings = 0
    total = 0.0
    for s in worlds:
        values = []
        for food in (robot.RED, robot.BLUE):
            per_half = robot.lifetime(rule, _geometry, task, s, food, world, evolve.TICKS[task])[0]
            halves += per_half[:, :2]
            firings += per_half[:, 2].sum()
            f, p = per_half[:, 0].sum(), per_half[:, 1].sum()
            values.append(f - evolve.POISON * p - evolve.FIRE_COST * per_half[:, 2].sum())
        total += sum(values) if task == "taste" else min(values)
    learning = learned(halves, 2 * len(worlds))
    score = total / len(worlds) + evolve.LEARN * learning
    (f1, p1), (f2, p2) = halves
    meals = halves.sum() / (2 * len(worlds))
    behaviour = (meals, (f2 - f1) / (f1 + f2 + 2), (p2 - p1) / (p1 + p2 + 2))
    return score, behaviour, learning, meals, halves


def learned(halves, lives: int) -> float:
    """The rise in food share from the first half of lives to the second; 0 unless both halves have enough meals."""
    (f1, p1), (f2, p2) = halves
    if min(f1 + p1, f2 + p2) < MIN_MEALS * lives:
        return 0.0
    return f2 / (f2 + p2) - f1 / (f1 + p1)


def niche(behaviour) -> tuple[int, int, int]:
    meals, food, poison = behaviour
    b = lambda x, lo, hi: int(np.clip((x - lo) / (hi - lo) * BINS, 0, BINS - 1))
    return b(meals, 0, MAX_MEALS), b(food, -CHANGE, CHANGE), b(poison, -CHANGE, CHANGE)


def random_genome(rng) -> np.ndarray:
    scale = np.exp(rng.uniform(np.log(0.1), np.log(3.0)))
    return np.r_[rng.normal(0, scale, cell.N_PARAMS), rng.uniform(0, TASTE_MAX, 2)]


def seeded_genomes() -> list[np.ndarray]:
    """Our evolved move and approach rules, with a few ways of tasting."""
    genomes = []
    for name in ("move", "approach"):
        path = evolve.KERNELS / f"{name}.json"
        if path.exists():
            for body in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
                genomes.append(np.r_[evolve.load(path), body])
    return genomes


def offspring(archive, rng) -> np.ndarray:
    elites = list(archive.values())
    a = elites[rng.integers(len(elites))]["genome"]
    sigma = np.exp(rng.uniform(np.log(0.005), np.log(0.5)))
    child = a + rng.normal(0, sigma, N_GENES)
    if rng.random() < 0.5:                      # along the line towards another elite
        b = elites[rng.integers(len(elites))]["genome"]
        child += rng.normal(0, 0.5) * (b - a)
    child[-2:] = np.clip(child[-2:], 0, TASTE_MAX)
    return child


def save_rule(path: Path, task: str, elite: dict) -> None:
    genome = elite["genome"]
    path.write_text(json.dumps({
        "task": task, "score": elite["score"], "learning": elite["learning"], "meals": elite["meals"],
        "validated": elite.get("validated"),
        "variables": cell.VARIABLES, "rule": genome[:cell.N_PARAMS].tolist(),
        "body": {"taste_contact": float(genome[-2]), "taste_centre": float(genome[-1])}}, indent=1))


def run(hours: float, task: str, resume: bool) -> None:
    evolve.KERNELS.mkdir(exist_ok=True)
    store = evolve.KERNELS / f"map_{task}.pkl"
    archive, evaluations = {}, 0
    if resume and store.exists():
        archive, evaluations = pickle.loads(store.read_bytes())
    rng = np.random.default_rng(evaluations)
    deadline = time.time() + hours * 3600
    started = time.time()
    batches = 0
    validated = -1.0
    with Pool() as pool:
        while time.time() < deadline:
            if evaluations < INITIAL:
                genomes = seeded_genomes() if evaluations == 0 else []
                genomes += [random_genome(rng) for _ in range(BATCH - len(genomes))]
            else:
                genomes = [offspring(archive, rng) for _ in range(BATCH)]
            for genome, (score, behaviour, learning, meals, _halves) in zip(
                    genomes, pool.map(evaluate, [(g_, task, WORLDS) for g_ in genomes])):
                key = niche(behaviour)
                if key not in archive or score > archive[key]["score"]:
                    archive[key] = {"genome": genome, "score": score, "behaviour": behaviour,
                                    "learning": learning, "meals": meals}
            evaluations += len(genomes)
            batches += 1
            best = max(archive.values(), key=lambda e: e["score"])
            learner = max(archive.values(), key=lambda e: e["learning"])
            if batches % VALIDATE_EVERY == 0 and learner["learning"] > 0:
                chunks = [VALIDATION[i::8] for i in range(8)]
                results = pool.map(evaluate, [(learner["genome"], task, c) for c in chunks])
                fresh = learned(sum(r[4] for r in results), 2 * len(VALIDATION))
                print(f"    validating the best learner ({learner['learning']:+.2f} on the fixed worlds): "
                      f"{fresh:+.2f} on fresh worlds", flush=True)
                if fresh > validated:
                    validated = fresh
                    save_rule(evolve.KERNELS / f"map_{task}_validated.json", task, dict(learner, validated=fresh))
            if batches % 10 == 0 or time.time() >= deadline:
                store.write_bytes(pickle.dumps((archive, evaluations)))
                save_rule(evolve.KERNELS / f"map_{task}.json", task, best)
                save_rule(evolve.KERNELS / f"map_{task}_learner.json", task, learner)
                print(f"{(time.time() - started) / 60:6.1f} min  {evaluations:7d} evaluations  "
                      f"{len(archive):3d}/{BINS ** 3} niches  best score {best['score']:6.2f} "
                      f"(meals {best['meals']:.1f}, learning {best['learning']:+.2f})  "
                      f"best learner {learner['learning']:+.2f} (meals {learner['meals']:.1f}, score "
                      f"{learner['score']:.2f})  best validated {validated:+.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hours", type=float, nargs="?", default=8.0)
    parser.add_argument("--task", choices=("taste", "choose"), default="taste")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.hours, args.task, args.resume)
