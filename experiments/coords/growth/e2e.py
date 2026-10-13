"""End to end, every cell on its own estimate: A (fixed K0, domains), B (fade, estimate), C1 (P1 at 1.6 x own
estimate; cells without one don't grow), C2 (P2, damped by P1, rate x C2BOOST), then coordinates (expH)."""
import sys, numpy as np
from sim import means, degree, unfolded
from stages import stage_b
import stages as AB
import expH
from expG import error
from classify import classify
from multiprocessing import Pool
HU, HV, FOLLOW, A = 0.25, 0.05, 1.2, 1.0
expH.NEAR, expH.MARGIN = 0.45, 0.05
RESET = True
CROSS = 1.0
def grow(n, rng, u, v, sigma, K, ticks, deg, cross=None, c=1.0, P=0.5):
    for t in range(ticks):
        go = rng.random((n, n)) < P
        mu, mv = means(u[..., None], deg)[..., 0], means(v[..., None], deg)[..., 0]
        D = v - u
        sat = 1 - D ** 2 / A ** 2 - (c * cross ** 2 / A ** 2 if cross is not None else 0)
        nu = u + HU * deg * (mu - u) + FOLLOW * K * D; nv = v + HV * deg * (mv - v) + K * D * sat
        u = np.where(go, nu + sigma * rng.standard_normal((n, n)), u); v = np.where(go, nv + sigma * rng.standard_normal((n, n)), v)
    return u, v
def job(a):
    n, s, sigma, boost = a
    rng = np.random.default_rng(s); deg = degree(n); true = HV * 2 * (1 - np.cos(np.pi / n))
    r = rng.standard_normal((n, n)); u, v = r.copy(), r.copy()
    u, v = grow(n, rng, u, v, sigma, AB.K0, int(200 * n * n), deg)
    u, v, est = stage_b(n, rng, u, v, sigma, deg, int(60 / (0.5 * true)))
    K = np.where(np.isfinite(est), 1.6 * est, 0.0)
    T = int(40 / (0.5 * true))                                   # (the observer's time budget for the growth passes)
    u, v = grow(n, rng, u, v, sigma, K, T, deg); P1 = v - u
    if RESET:
        r2 = rng.standard_normal((n, n)); u, v = u + r2 * 0, u + 0 * r2
        u, v = r2.copy(), r2.copy()
    u, v = grow(n, rng, u, v, sigma, K * boost, T, deg, cross=P1, c=CROSS); P2 = v - u
    z = expH.coords(P1, P2, rng, sigma, 40 * n * n)
    return classify(np.stack([P1, P2], -1)), error(z)
if __name__ == "__main__":
    from collections import Counter
    boost = float(sys.argv[1]) if len(sys.argv) > 1 else 1.15
    AB.AGREE = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    CROSS = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    for n in (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '16,24,32').split(',')):
        for sigma in (1e-3, 1e-2):
            with Pool(8) as pool:
                r = pool.map(job, [(n, s, sigma, boost) for s in range(16)])
            e = np.array([x[1] for x in r])
            print(f"n {n} noise {sigma}: corners {dict(Counter(x[0] for x in r))}; coordinates right (error < 0.1) {(e < 0.1).sum()}/16 (median error {np.median(e):.3f})", flush=True)
