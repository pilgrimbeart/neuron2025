"""H: corners hold x, y by the signs of P1, P2; edge cells hold whichever coordinate their neighbours' average puts
nearest 0 or 1 (at that value), averaging the other; inside cells average."""
import sys, numpy as np
MARGIN = 0.0           # ... and clearly nearer than the other coordinate is, by this much
NEAR = 0.25            # an edge cell holds a coordinate only when its neighbours' average is this near 0 or 1
from sim import *
from expE import run
from expG import error
def coords(P1, P2, rng, sigma, ticks, Pr=0.5):
    n = P1.shape[0]; deg = degree(n); edge, corner = deg == 3, deg == 2
    z = 0.5 + 0.01 * rng.standard_normal((n, n, 2))
    cx, cy = np.where(P1 > 0, 0.0, 1.0), np.where(P2 > 0, 0.0, 1.0)
    for t in range(ticks):
        m = means(z, deg)
        new = m.copy()
        dist = np.minimum(m, 1 - m)                                   # how near 0 or 1 each neighbours' average is
        hx = edge & (dist[..., 0] < dist[..., 1] - MARGIN) & (dist[..., 0] < NEAR); hy = edge & (dist[..., 1] < dist[..., 0] - MARGIN) & (dist[..., 1] < NEAR)
        new[..., 0] = np.where(hx, np.round(m[..., 0]), new[..., 0]); new[..., 1] = np.where(hy, np.round(m[..., 1]), new[..., 1])
        new[..., 0] = np.where(corner, cx, new[..., 0]); new[..., 1] = np.where(corner, cy, new[..., 1])
        new += sigma * rng.standard_normal(new.shape)
        go = rng.random((n, n)) < Pr
        z = np.where(go[..., None], new, z)
    return z
if __name__ == "__main__":
    n = int(sys.argv[1])
    for sigma in (1e-3, 1e-2):
        out = []
        for s in range(8):
            P1 = run(n, s, sigma, 1.7); P2 = run(n, s + 1000, sigma, 1.7, cross=P1, c=1.0, form='saturate')
            z = coords(P1, P2, np.random.default_rng(s + 5000), sigma, 40 * n * n)
            out.append((error(z), unfolded(z)[0]))
        print(f"H n {n} noise {sigma}: largest coordinate error {np.round([o[0] for o in out], 3)}; unfolded {sum(o[1] for o in out)}/8", flush=True)
