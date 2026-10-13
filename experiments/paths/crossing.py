"""Crossing paths, approach 1: a separate set of variables (wave and breadcrumbs) per path. Each path is its own
copy of the breadcrumb rule, with its own source and sink; they share only the cells. Measures: do both join, and
where do they cross."""
import sys, numpy as np, paths as P
def lives(channels, seed=0, dt=1.0):
    gens = [P.life(seed + 17 * i, dt, ends) for i, ends in enumerate(channels)]
    while True:
        out = [next(g) for g in gens]
        yield out[0][0], [o[1] for o in out], [o[2] for o in out]
if __name__ == "__main__":
    S = P.SIZE
    # two paths that must cross: top-left source to bottom-right sink, bottom-left source to top-right sink
    channels = [P.Ends(sources=((6, 3),), sinks=((S - 7, S - 4),)), P.Ends(sources=((S - 7, 3),), sinks=((6, S - 4),))]
    for t, paths, ends in lives(channels):
        if t >= 300: break
    for i, (p, e) in enumerate(zip(paths, ends)):
        print(f"path {i}: {P.report(p, e)}")
    both = (paths[0] > 0.5) & (paths[1] > 0.5)
    print("cells shared by both paths:", np.argwhere(both).tolist())
    import pygame
    pic = np.maximum(paths[0] * 1.0, paths[1] * 0.6)
    P.save_pictures([pic], sys.argv[1] if len(sys.argv) > 1 else "/tmp/cross.png", scale=6)
