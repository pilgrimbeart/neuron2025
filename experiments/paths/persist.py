"""Persistence: over a long life, how often the source is joined to a sink, and how busy the waves are; at 1000 an
event: none, the sink removed, a wall across the path, or the sink moved."""
import sys, numpy as np, paths as P
def run(event, persist, seed=0, dt=1.0, jitter=0.01, end=2000):
    P.PERSIST, P.JITTER = persist, jitter
    ends = P.Ends()
    lives = P.life(seed, dt, ends)
    joined, busy = {"before": [], "after": []}, {"before": [], "after": []}
    for t, p, now in lives:
        s = lives.gi_frame.f_locals["s"]
        phase = "before" if t < 1000 else "after"
        if t >= 500 and t % 5 == 0 and (t < 1000 or t >= 1300):        # judged after things have had time to settle
            joined[phase].append(any(P.route(p, now.sources[0], k) is not None for k in now.sinks) if now.sinks else None)
            busy[phase].append((s[P.EXCITED_LEFT] > 0).mean())
        if abs(t - 1000) < dt / 2:
            change = {"none": {}, "sink removed": {"sinks": ()}, "wall": {"walls": tuple((r, 24) for r in range(14, 35))},
                      "damage": {"walls": tuple((r, 24) for r in range(14, 35)), "cuts": tuple((r, 24) for r in range(14, 35))},
                      "sink moved": {"sinks": ((6, 24),)}}[event]
            t, p, now = lives.send(P.replace(now, **change))
        if t >= end - dt / 2:
            final_cells = int((p > 0.5).sum()); break
    f = lambda xs: "-" if not xs or xs[0] is None else f"{np.mean(xs):.2f}"
    return (f"joined {f(joined['before'])} -> {f(joined['after'])}, wave activity {np.mean(busy['before']):.3f} -> "
            f"{np.mean(busy['after']):.3f}, path cells at the end {final_cells}")
if __name__ == "__main__":
    for persist in (False, True):
        for event in ("none", "sink removed", "damage", "sink moved"):
            print(f"{'persist' if persist else 'plain  '} {event:13s}: {run(event, persist)}", flush=True)
