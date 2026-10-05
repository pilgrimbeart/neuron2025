"""MAP-Elites: a wide search for a cell rule that learns, keeping the best rule for each kind of behaviour rather
than only the best rule.

    python mapelites.py [HOURS] [--task taste|choose|discriminate] [--resume] [--seed KERNEL...] [--variables v,w,m,t]
                        [--freeze v,w]

A genome is a rule (cell.py) and two body genes: how strongly a taste kicks the cell where the block touched and
the middle of the disc (robot.World.taste_contact, taste_centre; 0..TASTE_MAX). Each genome lives in WORLDS (or TASK_WORLDS[task]) fixed
worlds, each once with red as food and once with blue (in taste, the one red block is food, then poison), so every
evaluation is deterministic and an elite can't hold its place by luck; and it lives them at each tick size in
evolve.DTS, scoring the worse, so that no rule can rely on the tick size.

Behaviour, 3 axes of BINS bins each:
  meals     meals per life, 0..MAX_MEALS[task]
  food      how eating food changes from the first third of a life to the last, (F3 - F1) / (F1 + F3 + 2)
  poison    the same for poison, (P3 - P1) / (P1 + P3 + 2)
A learner sits where poison falls and food doesn't; eating everything, in the middle. The log's "spikes" is the
sheet's spikes per cell per 100 units of time (activity above evolve.SPARSE_RATE costs score). Within a cell the higher score
wins, scored as in evolve.py, except that learning (the rise in food share from the first third of lives to the
last) only counts when both have more than MIN_MEALS meals per life, so that stopping eating can't pass for
learning. Offspring: a random elite, mutated at a random scale (up to REACH), or crossed with another along the line
between them.
Racing: each candidate is lived at the tick sizes one at a time, coarsest (cheapest) first. Its score can only fall
as tick sizes are added (the score is the worst over them), so after each, only candidates that already beat their
niche's elite, or land in an empty niche, go on. That rejects nothing that could have won (bar a candidate whose
niche moves as tick sizes are added), and typically only a few percent go past the first.

--variables and --freeze limit which rule coefficients may change (evolve.free_mask), the fixed part taken from the
first --seed kernel: a stage can add variables for a new challenge while what already works stays.

Without --seed, the first INITIAL candidates are random rules. With it, they are the saved rules (a kernel, .json, with
its body genes, or every elite of another search's archive, .pkl) and variations of them, and mutation stays small
(SEEDED_REACH): a working robot survives small changes to many terms but not large ones (with 622 free terms, the
24x24 forager stops eating at a spread of about 0.1).

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
TASK_WORLDS = {"discriminate": range(16)}     # long lives (about 100 meals) are less noisy: fewer worlds suffice
VALIDATION = range(10**6, 10**6 + 64)
VALIDATE_EVERY = 50
MIN_MEALS = 0.25
BINS = 8
MAX_MEALS = {"taste": 8.0, "choose": 8.0, "discriminate": 80.0}   # the meals axis's range, per task
CHANGE = 0.6                # the food and poison axes span -CHANGE..CHANGE
TASTE_MAX = 1.5
BATCH = 64
INITIAL = 256
REACH = 0.5                 # the largest mutation scale
SEEDED_REACH = 0.1          # the same, near saved rules
SMALLEST = 0.005            # the smallest
N_GENES = cell.N_PARAMS + 2



def evaluate(args):
    """(score, behaviour (meals, food change, poison change), learning, meals per life, parts, spike rate) for a
    genome lived at each tick size in dts: `combine` of `live_at` for each."""
    genome, task, worlds, dts = args
    return combine([live_at((genome, task, worlds, dt)) for dt in dts], len(worlds))


def live_at(args):
    """A genome lived in each world (red as food, then blue) at one tick size: its score (the mean of the worlds'
    values, plus the learning bonus), learning, the (food, poison) eaten in each part of its lives (PARTS, 2), and its
    spike rate (spikes per cell per unit time)."""
    genome, task, worlds, dt = args
    world = robot.World(taste_contact=float(genome[-2]), taste_centre=float(genome[-1]))
    a = evolve.assess(genome[:cell.N_PARAMS], task, worlds, dt, world)
    parts = a["parts"].sum(axis=0)
    learning = evolve.learned(parts, 2 * len(worlds), MIN_MEALS)
    return {"score": float(a["values"].mean()) + evolve.LEARN * learning, "learning": learning, "parts": parts,
            "spikes": float(a["spikes"].mean())}


def combine(at, worlds: int):
    """The tick sizes' results as one: score and learning the worse over them (so no rule can rely on the tick
    size); behaviour and meals pooled over them; parts (len(at), PARTS, 2); spike rate the mean."""
    parts = np.array([r["parts"] for r in at])
    pooled = parts.sum(axis=0)
    (f1, p1), (f2, p2) = pooled[0], pooled[-1]
    meals = parts.sum() / (2 * worlds * len(at))
    behaviour = (meals, (f2 - f1) / (f1 + f2 + 2), (p2 - p1) / (p1 + p2 + 2))
    return (min(r["score"] for r in at), behaviour, min(r["learning"] for r in at), meals, parts,
            float(np.mean([r["spikes"] for r in at])))


def could_win(at, archive, task: str, worlds: int) -> bool:
    """Whether a candidate lived at some tick sizes so far could still win its niche: its score can only fall as
    tick sizes are added (it is the worse over them), so only if it already beats the niche's elite, or the niche is
    empty."""
    score, behaviour = combine(at, worlds)[:2]
    key = niche(behaviour, task)
    return key not in archive or score > archive[key]["score"]


def niche(behaviour, task: str) -> tuple[int, int, int]:
    meals, food, poison = behaviour
    b = lambda x, lo, hi: int(np.clip((x - lo) / (hi - lo) * BINS, 0, BINS - 1))
    return b(meals, 0, MAX_MEALS[task]), b(food, -CHANGE, CHANGE), b(poison, -CHANGE, CHANGE)


def random_genome(rng) -> np.ndarray:
    scale = np.exp(rng.uniform(np.log(0.1), np.log(3.0)))
    return np.r_[rng.normal(0, scale, cell.N_PARAMS), rng.uniform(0, TASTE_MAX, 2)]


def variant(seeds, rng, reach: float) -> np.ndarray:
    """A saved rule, varied at a random scale up to reach."""
    genome = seeds[rng.integers(len(seeds))].copy()
    genome[:cell.N_PARAMS] += rng.normal(0, np.exp(rng.uniform(np.log(SMALLEST), np.log(reach))), cell.N_PARAMS)
    return genome


def offspring(archive, rng, reach: float) -> np.ndarray:
    elites = list(archive.values())
    a = elites[rng.integers(len(elites))]["genome"]
    sigma = np.exp(rng.uniform(np.log(SMALLEST), np.log(reach)))
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


def from_kernels(paths) -> list[np.ndarray]:
    """Saved rules as genomes: each kernel, and every elite of an archive."""
    genomes = []
    for path in paths:
        if str(path).endswith(".pkl"):
            archive, _ = pickle.loads(Path(path).read_bytes())
            genomes += [elite["genome"].copy() for elite in archive.values()]
            continue
        body = evolve.load_world(path)
        genome = np.r_[evolve.load(path), body.taste_contact, body.taste_centre]
        genomes.append(genome)
    return genomes


def run(hours: float, task: str, resume: bool, seeds=(), free=None) -> None:
    """free (evolve.free_mask) says which rule coefficients the search may change; the rest stay as in the first
    seed kernel (the body genes are always free)."""
    worlds = TASK_WORLDS.get(task, WORLDS)
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
    saved = from_kernels(seeds)
    extra = list(saved)                                 # the saved rules themselves go first
    reach = SEEDED_REACH if saved else REACH
    free = np.r_[np.ones(cell.N_PARAMS, dtype=bool) if free is None else free, True, True]
    base = saved[0] if saved else np.zeros(N_GENES)
    held = lambda g: np.where(free, g, base)            # every genome keeps the fixed part
    with Pool() as pool:
        while time.time() < deadline:
            if evaluations < INITIAL:
                genomes = [variant(saved, rng, reach) if saved else random_genome(rng) for _ in range(BATCH)]
            else:
                genomes = [offspring(archive, rng, reach) for _ in range(BATCH)]
            genomes = [held(g) for g in genomes]
            if extra:
                genomes, extra = extra[:BATCH] + genomes[len(extra[:BATCH]):], extra[BATCH:]
            first = pool.map(live_at, [(g_, task, worlds, evolve.DTS[0]) for g_ in genomes])
            racing = [(g_, [r]) for g_, r in zip(genomes, first)]
            for dt in evolve.DTS[1:]:           # stage by stage: only candidates that could still win go on
                racing = [(g_, at) for g_, at in racing if could_win(at, archive, task, len(worlds))]
                more = pool.map(live_at, [(g_, task, worlds, dt) for g_, _ in racing])
                racing = [(g_, at + [r]) for (g_, at), r in zip(racing, more)]
            raced += len(genomes)
            passed += len(racing)
            for genome, at in racing:
                score, behaviour, learning, meals, _parts, rate = combine(at, len(worlds))
                key = niche(behaviour, task)
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
                fresh = min(evolve.learned(h, 2 * len(VALIDATION), MIN_MEALS) for h in pooled)
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
    parser.add_argument("--task", choices=("taste", "choose", "discriminate"), default="taste")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", nargs="+", default=[], help="saved rules to add to the first batch")
    parser.add_argument("--variables", default=",".join(cell.VARIABLES), help="the variables in use, e.g. v,w")
    parser.add_argument("--freeze", default="", help="variables whose rule (among themselves) stays as the first seed's")
    args = parser.parse_args()
    run(args.hours, args.task, args.resume, args.seed,
        evolve.free_mask(args.variables.split(","), [x for x in args.freeze.split(",") if x]))
