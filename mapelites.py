"""MAP-Elites: a wide search for a cell rule that learns, keeping the best rule for each kind of behaviour rather
than only the best rule.

    python mapelites.py [HOURS] [--task taste|choose] [--resume] [--seed KERNEL...]

A genome is a rule (cell.py) and two body genes: how strongly a taste kicks the cell where the block touched and
the middle of the disc (robot.World.taste_contact, taste_centre; 0..TASTE_MAX). Each genome lives in WORLDS fixed
worlds, each once with red as food and once with blue (in taste, the one red block is food, then poison), so every
evaluation is deterministic and an elite can't hold its place by luck; and it lives them at each tick size in
evolve.DTS, scoring the worse, so that no rule can rely on the tick size.

Behaviour, 3 axes of BINS bins each:
  meals     meals per life, 0..MAX_MEALS
  food      how eating food changes from the first half of a life to the second, (F2 - F1) / (F1 + F2 + 2)
  poison    the same for poison, (P2 - P1) / (P1 + P2 + 2)
A learner sits where poison falls and food doesn't; eating everything, in the middle. The log's "spikes" is the
sheet's spikes per cell per 100 units of time (activity above evolve.SPARSE_RATE costs score). Within a cell the higher score
wins, scored as in evolve.py, except that learning (the rise in food share from the first half of lives to the
second) only counts when both halves have at least MIN_MEALS meals per life, so that stopping eating can't pass for
learning. Offspring: a random elite, mutated at a random scale, or crossed with another along the line between them.
Racing: each candidate is first lived at the first tick size only (a third of the work). Its score can only fall
when the other tick sizes are added (the score is the worse over them), so only a candidate that already beats its
niche's elite, or lands in an empty niche, is lived at the others. That rejects nothing that could have won (bar a
candidate whose niche moves when the other tick sizes are added), and typically only a few percent go on.

--seed adds saved rules to the first batches: a kernel (.json, with its body genes) with SEED_VARIANTS small
variations, or every elite of another search's archive (.pkl). A way to bring in rules from another search, such as
one on an easier task.

The archive is saved to kernels/map_TASK.pkl, the best-scoring elite to kernels/map_TASK.json and the best learner to
kernels/map_TASK_learner.json. Fixed worlds let flukes in, so every VALIDATE_EVERY batches the best learner is lived
again in VALIDATION fresh worlds, at each tick size in evolve.CHECK_DTS (finer still); one that learns there too, at
every one of them, is saved to kernels/map_TASK_validated.json.
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
SEED_VARIANTS = 7
SEED_SPREAD = 0.02
N_GENES = cell.N_PARAMS + 2



def evaluate(args):
    """(score, behaviour (meals, food change, poison change), learning, meals per life, halves, spike rate) for a
    genome lived at each tick size in dts: `combine` of `live_at` for each."""
    genome, task, worlds, dts = args
    return combine([live_at((genome, task, worlds, dt)) for dt in dts], len(worlds))


def live_at(args):
    """A genome lived in each world (red as food, then blue) at one tick size: its score (the mean of the worlds'
    values, plus the learning bonus), learning, the (food, poison) eaten in each half of its lives (2, 2), and its
    spike rate (spikes per cell per unit time)."""
    genome, task, worlds, dt = args
    world = robot.World(taste_contact=float(genome[-2]), taste_centre=float(genome[-1]))
    a = evolve.assess(genome[:cell.N_PARAMS], task, worlds, dt, world)
    halves = a["halves"].sum(axis=0)
    learning = learned(halves, 2 * len(worlds))
    return {"score": float(a["values"].mean()) + evolve.LEARN * learning, "learning": learning, "halves": halves,
            "spikes": float(a["spikes"].mean())}


def combine(at, worlds: int):
    """The tick sizes' results as one: score and learning the worse over them (so no rule can rely on the tick
    size); behaviour and meals pooled over them; halves (len(at), 2, 2); spike rate the mean."""
    halves = np.array([r["halves"] for r in at])
    (f1, p1), (f2, p2) = halves.sum(axis=0)
    meals = halves.sum() / (2 * worlds * len(at))
    behaviour = (meals, (f2 - f1) / (f1 + f2 + 2), (p2 - p1) / (p1 + p2 + 2))
    return (min(r["score"] for r in at), behaviour, min(r["learning"] for r in at), meals, halves,
            float(np.mean([r["spikes"] for r in at])))


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
        "spikes_per_cell_per_100": 100 * elite["spikes"],
        "validated": elite.get("validated"),
        "variables": cell.VARIABLES, "rule": genome[:cell.N_PARAMS].tolist(),
        "body": {"taste_contact": float(genome[-2]), "taste_centre": float(genome[-1])}}, indent=1))


def from_kernels(paths, rng) -> list[np.ndarray]:
    """Saved rules as genomes: each kernel with SEED_VARIANTS small variations of its rule, and every elite of an
    archive."""
    genomes = []
    for path in paths:
        if str(path).endswith(".pkl"):
            archive, _ = pickle.loads(Path(path).read_bytes())
            genomes += [elite["genome"].copy() for elite in archive.values()]
            continue
        body = evolve.load_world(path)
        genome = np.r_[evolve.load(path), body.taste_contact, body.taste_centre]
        genomes.append(genome)
        for _ in range(SEED_VARIANTS):
            variant = genome.copy()
            variant[:cell.N_PARAMS] += rng.normal(0, SEED_SPREAD, cell.N_PARAMS)
            genomes.append(variant)
    return genomes


def run(hours: float, task: str, resume: bool, seeds=()) -> None:
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
    raced = passed = 0
    extra = from_kernels(seeds, rng)
    with Pool() as pool:
        while time.time() < deadline:
            if evaluations < INITIAL:
                genomes = seeded_genomes() if evaluations == 0 else []
                genomes += [random_genome(rng) for _ in range(BATCH - len(genomes))]
            else:
                genomes = [offspring(archive, rng) for _ in range(BATCH)]
            if extra:
                genomes, extra = extra[:BATCH] + genomes[len(extra[:BATCH]):], extra[BATCH:]
            first = pool.map(live_at, [(g_, task, WORLDS, evolve.DTS[0]) for g_ in genomes])
            racing = []                         # candidates whose first tick size already beats their niche's elite
            for g_, r in zip(genomes, first):
                key = niche(combine([r], len(WORLDS))[1])
                if key not in archive or r["score"] > archive[key]["score"]:
                    racing.append((g_, r))
            rest = pool.map(live_at, [(g_, task, WORLDS, dt) for g_, _ in racing for dt in evolve.DTS[1:]])
            others = len(evolve.DTS) - 1
            raced += len(genomes)
            passed += len(racing)
            for c, (genome, r) in enumerate(racing):
                score, behaviour, learning, meals, _halves, rate = combine(
                    [r] + rest[c * others:(c + 1) * others], len(WORLDS))
                key = niche(behaviour)
                if key not in archive or score > archive[key]["score"]:
                    archive[key] = {"genome": genome, "score": score, "behaviour": behaviour,
                                    "learning": learning, "meals": meals, "spikes": rate}
            evaluations += len(genomes)         # candidates tried
            batches += 1
            best = max(archive.values(), key=lambda e: e["score"])
            learner = max(archive.values(), key=lambda e: e["learning"])
            if batches % VALIDATE_EVERY == 0 and learner["learning"] > 0:
                chunks = [VALIDATION[i::8] for i in range(8)]
                results = pool.map(evaluate, [(learner["genome"], task, c, evolve.CHECK_DTS) for c in chunks])
                pooled = sum(r[4] for r in results)
                fresh = min(learned(h, 2 * len(VALIDATION)) for h in pooled)
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
                      f"(meals {best['meals']:.1f}, learning {best['learning']:+.2f}, spikes {100 * best['spikes']:.1f})  "
                      f"best learner {learner['learning']:+.2f} (meals {learner['meals']:.1f}, score "
                      f"{learner['score']:.2f}, spikes {100 * learner['spikes']:.1f})  best validated {validated:+.2f}  "
                      f"raced {passed}/{raced} through",
                      flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hours", type=float, nargs="?", default=8.0)
    parser.add_argument("--task", choices=("taste", "choose"), default="taste")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", nargs="+", default=[], help="saved rules to add to the first batch")
    args = parser.parse_args()
    run(args.hours, args.task, args.resume, args.seed)
