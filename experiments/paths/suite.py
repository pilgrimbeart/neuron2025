"""The test suite for path-making: join, move, remove, several ends; noise on and off; seeds; tick sizes."""
import sys, numpy as np, paths as P
SCENES = {
    "one pair, sink moves": (P.Ends(), lambda n: P.replace(n, sinks=((6, 24),))),
    "3 sources 2 sinks, one sink removed": (P.Ends(sources=((8, 45), (24, 45), (40, 45)), sinks=((12, 2), (36, 2))), lambda n: P.replace(n, sinks=n.sinks[:1])),
    "3 sources 2 sinks, all sinks removed": (P.Ends(sources=((8, 45), (24, 45), (40, 45)), sinks=((12, 2), (36, 2))), lambda n: P.replace(n, sinks=())),
    "scattered, source moves": (P.Ends(sources=((5, 40), (30, 44), (44, 20)), sinks=((4, 4), (25, 10), (40, 40))), lambda n: P.replace(n, sources=((20, 24),) + n.sources[1:])),
}
def joined_fraction(p, now):
    """Every source should reach some sink: the fraction that do."""
    if not now.sinks:
        return None
    return np.mean([any(P.route(p, a, b) is not None for b in now.sinks) for a in now.sources])
def run(seed=0, dt=1.0, jitter=0.01, change_at=300, end=600):
    P.JITTER = jitter
    rows = []
    for name, (ends, change) in SCENES.items():
        lives = P.life(seed, dt, ends)
        marks = {}
        for t, p, now in lives:
            if abs(t - change_at) < dt / 2:
                marks["before"] = (joined_fraction(p, now), int((p > 0.5).sum()))
                t, p, now = lives.send(change(now))
            if abs(t - (change_at + 150)) < dt / 2:
                marks["+150"] = (joined_fraction(p, now), int((p > 0.5).sum()))
            if t >= end - dt / 2:
                marks["end"] = (joined_fraction(p, now), int((p > 0.5).sum()))
                break
        rows.append(f"  {name:40s} " + "  ".join(f"{k}: {'-' if f is None else f'{f:.2f}'} joined, {c} cells" for k, (f, c) in marks.items()))
    return "\n".join(rows)
if __name__ == "__main__":
    for seed, dt, jitter in ((0, 1.0, 0.0), (0, 1.0, 0.01), (1, 1.0, 0.01), (0, 0.5, 0.01)):
        print(f"seed {seed}, dt {dt}, jitter {jitter}:\n" + run(seed, dt, jitter), flush=True)
