"""E: grow the large pattern. u (fast copy) averages and follows v; v (slow copy) averages and grows its difference
from u (rate K), saturating at size A. Growth rate set by the observer here: FACTOR x the slowest fade rate of v."""
import sys, numpy as np
from sim import *
def run(n, seed, sigma, factor=1.7, HU=0.25, HV=0.05, A=1.0, ticks=None, P=0.5, follow=1.2, cross=None, c=2.0, theta=0.05, form='growth'):
    rng = np.random.default_rng(seed); deg = degree(n)
    lam = 2 * (1 - np.cos(np.pi / n)); kc = HV * lam; K = factor * kc; KU = follow * K
    r = rng.standard_normal((n, n)); u, v = r.copy(), r.copy()
    ticks = ticks or int(40 / (0.5 * kc))
    for t in range(ticks):
        mu, mv = means(u[..., None], deg)[..., 0], means(v[..., None], deg)[..., 0]
        D = v - u
        damp = np.where(np.abs(D) < theta * A, c * cross ** 2 / A ** 2, 0.0) if cross is not None else 0.0
        sat = (1 - D ** 2 / A ** 2) * (1 - damp) if form == 'growth' else 1 - D ** 2 / A ** 2 - (c * cross ** 2 / A ** 2 if cross is not None else 0)
        nu = u + HU * deg * (mu - u) + KU * D + sigma * rng.standard_normal((n, n))
        nv = v + HV * deg * (mv - v) + K * D * sat + sigma * rng.standard_normal((n, n))
        go = rng.random((n, n)) < P
        u, v = np.where(go, nu, u), np.where(go, nv, v)
    return v - u
def shape(D):
    """How much of the pattern is one straight west-east or south-north slow pattern (cos), 0..1."""
    n = D.shape[0]; x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    X, Y = np.tile(x, (n, 1)), np.tile(x[:, None], (1, n))
    d = D - D.mean()
    best = max(abs((d * X).sum()) / np.sqrt((d ** 2).sum() * (X ** 2).sum()), abs((d * Y).sum()) / np.sqrt((d ** 2).sum() * (Y ** 2).sum()))
    slow = np.sqrt((((d * X).sum() / (X ** 2).sum()) ** 2 * (X ** 2).sum() + ((d * Y).sum() / (Y ** 2).sum()) ** 2 * (Y ** 2).sum()) / (d ** 2).sum())
    return best, slow
if __name__ == "__main__":
    n = int(sys.argv[1])
    for sigma in (1e-3, 1e-2):
        for factor in (1.5, 1.9):
            out = []
            for s in range(6):
                D = run(n, s, sigma, factor)
                best, slow = shape(D)
                out.append((np.abs(D).max(), best, slow))
            print(f"E n {n} noise {sigma} rate {factor}x: size {np.median([o[0] for o in out]):.2f}; share in the slow pair {np.median([o[2] for o in out]):.3f}; share in one straight pattern {np.median([o[1] for o in out]):.3f}", flush=True)
