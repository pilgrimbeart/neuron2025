"""The whole chain with corner flags: flags (u), labels, then coordinates (u = x, v = y) with straightening edges.
Shared: u, v (side neighbours' averages only). Private: corners' state machines; everyone's own clock."""
import sys, numpy as np
from sim import means, degree, unfolded
import flags2 as F
from multiprocessing import Pool
NEAR, MARGIN, HOLD, STILL = 0.45, 0.05, 3.0, 0.03
def error(z):
    n = z.shape[0]; i = np.arange(n) / (n - 1); X, Y = np.tile(i, (n, 1)), np.tile(i[:, None], (1, n))
    best = 9
    for a, b in ((z[..., 0], z[..., 1]), (z[..., 1], z[..., 0])):
        for fa in (lambda q: q, lambda q: 1 - q):
            for fb in (lambda q: q, lambda q: 1 - q):
                best = min(best, max(np.abs(fa(a) - X).max(), np.abs(fb(b) - Y).max()))
    return best
def job(a):
    n, seed, sigma, maxticks, extra = a
    rng = np.random.default_rng(seed); deg = degree(n); edge = deg == 3
    cs = [(0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 0)]
    u = np.zeros((n, n)); v = np.full((n, n), 0.5); age = np.zeros((n, n)); passed = np.zeros((n, n), bool); slow = np.zeros((n, n)); settled = np.zeros((n, n), bool)
    C = [F.Corner() for _ in cs]; done = [None] * 4; finish = None
    for t in range(maxticks):
        go = rng.random((n, n)) < 0.5
        age = age + go
        mu, mv = means(u[..., None], deg)[..., 0], means(v[..., None], deg)[..., 0]
        nu, nv = mu.copy(), mv.copy()
        # edges straighten, but only once the corners' coordinates have reached them (v moved off 0.5)
        passed |= mu > F.FLAG / 2                                        # the flags have reached me
        slow = np.where(go, slow + 0.01 * (mu - slow), slow)                 # a slow private average of what I see
        live = edge & passed & (mu < 1.5) & (np.abs(mu - slow) < STILL)      # ... and gone, and u has stopped falling
        dx, dy = np.minimum(np.abs(mu), np.abs(1 - mu)), np.minimum(np.abs(mv), np.abs(1 - mv))
        hx = live & (dx < dy - MARGIN) & (dx < NEAR); hy = live & (dy < dx - MARGIN) & (dy < NEAR)
        nu = np.where(hx, np.round(mu), nu); nv = np.where(hy, np.round(mv), nv)
        for k, c in enumerate(cs):
            x = C[k]
            if done[k] is None:
                if go[c]: x.see(age[c], mu[c], rng)
                lab = x.label()
                if lab is not None and x.start is None and age[c] - x.raised > HOLD * x.settle:
                    done[k] = lab        # labelled, and my flag has been up long enough for the far corners to record it
                nu[c] = F.FLAG if x.raised is not None else 0.0; nv[c] = 0.5
            if done[k] is not None:
                nu[c], nv[c] = done[k]
        nu += sigma * rng.standard_normal((n, n)); nv += sigma * rng.standard_normal((n, n))
        u, v = np.where(go, nu, u), np.where(go, nv, v)
        if finish is None and None not in done: finish = t
        if finish is not None and t > finish + extra * n * n: break
    labs = done
    square = None not in labs and len(set(labs)) == 4 and all(abs(labs[k][0] - labs[(k + 1) % 4][0]) + abs(labs[k][1] - labs[(k + 1) % 4][1]) == 1 for k in range(4))
    z = np.stack([u, v], -1)
    return square, error(z), unfolded(z)[0], finish
if __name__ == "__main__":
    if len(sys.argv) > 2: F.SETTLE, F.WAIT = float(sys.argv[2]), float(sys.argv[3])
    for n in (int(x) for x in sys.argv[1].split(",")):
        for sigma in (1e-2,):
            with Pool(8) as pool:
                r = pool.map(job, [(n, s, sigma, 8_000_000, 60) for s in range(8)])
            e = np.array([x[1] for x in r])
            print(f"n {n} noise {sigma}: corner labels a proper square {sum(x[0] for x in r)}/8; coordinates right (error < 0.1) {(e < 0.1).sum()}/8, errors {np.round(e, 3)}; labelled by tick {[x[3] for x in r]}", flush=True)
