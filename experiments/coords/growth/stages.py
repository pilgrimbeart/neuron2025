"""A: grow at fixed K0 (O(1) domains). B: K = 0, fade; each cell estimates the fade rate from v - u at doubling ages
since B began (ratio of successive changes), trusting two that agree within STEADY. C: grow at 1.6 x its estimate."""
import sys, numpy as np
from sim import means, degree
from multiprocessing import Pool
HU, HV, FOLLOW, A = 0.25, 0.05, 1.2, 1.0
K0, GROWA, STEADY, FACTOR = 0.04, 200.0, 0.1, 1.6
AGREE = 2
MINFADE = 4.0
def stage_b(n, rng, u, v, sigma, deg, ticks, P=0.5):
    age = np.zeros((n, n)); nxt = np.full((n, n), 16.0)
    d4, d2, last, last2 = (np.full((n, n), np.nan) for _ in range(4)); got = np.full((n, n), np.nan); gotage = np.full((n, n), np.nan)
    for t in range(ticks):
        go = rng.random((n, n)) < P
        mu, mv = means(u[..., None], deg)[..., 0], means(v[..., None], deg)[..., 0]
        u = np.where(go, u + HU * deg * (mu - u) + sigma * rng.standard_normal((n, n)), u)
        v = np.where(go, v + HV * deg * (mv - v) + sigma * rng.standard_normal((n, n)), v)
        age = age + go
        at = go & (age >= nxt); D = v - u
        ratio = (D - d2) / (d2 - d4); y = (-1 + np.sqrt(np.maximum(1 + 4 * ratio, 0))) / 2
        k = np.where((ratio > 0) & (y > 0) & (y < 1), -4 * np.log(np.maximum(y, 1e-300)) / age, np.nan)
        agree = np.isfinite(k) & np.isfinite(last) & (np.abs(k - last) < STEADY * np.abs(k))
        if AGREE >= 3:
            agree = agree & np.isfinite(last2) & (np.abs(k - last2) < STEADY * np.abs(k))
        ok = at & agree & np.isnan(got) & (age * k >= MINFADE)       # the fade must have run a few of its own time units
        got = np.where(ok, k, got); gotage = np.where(ok, age, gotage); last2 = np.where(at, last, last2); last = np.where(at, k, last)
        d4 = np.where(at, d2, d4); d2 = np.where(at, D, d2); nxt = np.where(at, nxt * 2, nxt)
        if np.isfinite(got).mean() > 0.95: break
    stage_b.age = gotage
    return u, v, got
def job(a):
    n, s, sigma = a
    rng = np.random.default_rng(s); deg = degree(n); true = HV * 2 * (1 - np.cos(np.pi / n))
    r = rng.standard_normal((n, n)); u, v = r.copy(), r.copy()
    for t in range(int(GROWA * n * n)):
        go = rng.random((n, n)) < 0.5
        mu, mv = means(u[..., None], deg)[..., 0], means(v[..., None], deg)[..., 0]
        D = v - u
        nu = u + HU * deg * (mu - u) + FOLLOW * K0 * D; nv = v + HV * deg * (mv - v) + K0 * D * (1 - D ** 2 / A ** 2)
        u = np.where(go, nu + sigma * rng.standard_normal((n, n)), u); v = np.where(go, nv + sigma * rng.standard_normal((n, n)), v)
    u, v, got = stage_b(n, rng, u, v, sigma, deg, int(60 / (0.5 * true)))
    return got / true
if __name__ == "__main__":
    for n in (16, 24, 32):
        with Pool(8) as pool:
            r = pool.map(job, [(n, s, 1e-2) for s in range(8)])
        allk = np.concatenate([x.ravel() for x in r])
        print(f"n {n} noise 1e-2: cells with an estimate {np.isfinite(allk).mean():.0%}; estimate / true: median {np.nanmedian(allk):.2f}, 10-90% {np.nanpercentile(allk, 10):.2f}-{np.nanpercentile(allk, 90):.2f}; per run medians {[round(float(np.nanmedian(x)), 2) for x in r]}", flush=True)
