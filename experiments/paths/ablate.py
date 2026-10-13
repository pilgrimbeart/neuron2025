"""Score variants of the rule on the suite, in parallel: each variant is a dict of module constants to set."""
import sys, json, numpy as np
from multiprocessing import Pool
SCENES = ["one pair, sink moves", "3 sources 2 sinks, one sink removed", "3 sources 2 sinks, all sinks removed", "scattered, source moves"]
CONDITIONS = [(seed, dt, 0.01) for seed in range(4) for dt in (1.0, 0.5)]
def one(args):
    variant, scene, (seed, dt, jitter) = args
    import paths as P
    sys.path.insert(0, "/tmp/claude-1000/-home-pilgrim-git-neuron2025/7922b27f-2f98-45d8-802c-92ffd2bc4962/scratchpad")
    import suite
    for k, v in variant.items():
        setattr(P, k, v)
    P.JITTER = variant.get("JITTER", jitter)
    ends, change = suite.SCENES[scene]
    lives = P.life(seed, dt, ends)
    res = {}
    for t, p, now in lives:
        if abs(t - 300) < dt / 2:
            res["before"] = (suite.joined_fraction(p, now), int((p > 0.5).sum()))
            t, p, now = lives.send(change(now))
        if abs(t - 450) < dt / 2:
            res["+150"] = (suite.joined_fraction(p, now), int((p > 0.5).sum()))
        if t >= 600 - dt / 2:
            res["end"] = (suite.joined_fraction(p, now), int((p > 0.5).sum()))
            break
    return res
def score(variant, pool):
    jobs = [(variant, sc, c) for sc in SCENES for c in CONDITIONS]
    out = pool.map(one, jobs)
    joined = [r[k][0] for r in out for k in ("before", "end") if r[k][0] is not None]
    cells = [r[k][1] for r, (v, sc, c) in zip(out, jobs) for k in ("before", "end") if "all sinks" not in sc]
    cleared = [r["end"][1] for r, (v, sc, c) in zip(out, jobs) if "all sinks" in sc]
    return f"joined {np.mean(joined):.2f} (min {np.min(joined):.2f}), cells {np.mean(cells):5.1f}, cleared to {np.mean(cleared):.1f}"
if __name__ == "__main__":
    variants = json.loads(sys.argv[1])
    with Pool(8) as pool:
        for name, v in variants.items():
            print(f"{name:40s} {score(v, pool)}", flush=True)
