"""G: the whole chain with the growth rate set by the observer. Pass 1 grows P1; pass 2 grows P2 (damped by P1);
pass 3: u, v become x, y: an edge cell holds the coordinate whose pattern is larger in size at it (0 if positive,
1 if negative), a corner both, and everyone else averages. Noise in every update of every pass."""
import sys, numpy as np
from sim import *
from expE import run
def coords(P1, P2, rng, sigma, ticks, Pr=0.5):
    n = P1.shape[0]; deg = degree(n); edge, corner = deg == 3, deg == 2
    holdx = corner | (edge & (np.abs(P1) >= np.abs(P2))); holdy = corner | (edge & (np.abs(P2) > np.abs(P1)))
    tx, ty = np.where(P1 > 0, 0.0, 1.0), np.where(P2 > 0, 0.0, 1.0)
    z = 0.5 + 0.01 * rng.standard_normal((n, n, 2))
    for t in range(ticks):
        m = means(z, deg)
        new = m.copy()
        new[..., 0] = np.where(holdx, tx, new[..., 0]); new[..., 1] = np.where(holdy, ty, new[..., 1])
        new += sigma * rng.standard_normal(new.shape)
        go = rng.random((n, n)) < Pr
        z = np.where(go[..., None], new, z)
    return z
def error(z):
    n = z.shape[0]; i = np.arange(n) / (n - 1); X, Y = np.tile(i, (n, 1)), np.tile(i[:, None], (1, n))
    best = 9
    for a, b in ((z[..., 0], z[..., 1]), (z[..., 1], z[..., 0])):
        for fa in (lambda q: q, lambda q: 1 - q):
            for fb in (lambda q: q, lambda q: 1 - q):
                best = min(best, max(np.abs(fa(a) - X).max(), np.abs(fb(b) - Y).max()))
    return best
def chain(n, seed, sigma, factor=1.7, c=1.0):
    rng = np.random.default_rng(seed + 5000)
    P1 = run(n, seed, sigma, factor); P2 = run(n, seed + 1000, sigma, factor, cross=P1, c=c)
    return coords(P1, P2, rng, sigma, int(30 * n * n))
if __name__ == "__main__":
    n = int(sys.argv[1])
    for sigma in (1e-3, 1e-2):
        res = [chain(n, s, sigma) for s in range(8)]
        e = [error(z) for z in res]
        print(f"G n {n} noise {sigma}: largest coordinate error per run {np.round(e, 3)}; unfolded {sum(unfolded(z)[0] for z in res)}/8", flush=True)
