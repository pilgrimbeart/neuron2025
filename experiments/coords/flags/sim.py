"""A fast simulator for two-number coordinate rules: side neighbours only (means), own neighbour count d, private
memory, random updates (each cell with chance P per tick), ongoing noise SIGMA in every update."""
import numpy as np
def means(z, deg):
    p = np.pad(z, ((1, 1), (1, 1), (0, 0)))
    s = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return s / deg[..., None]
def degree(n):
    one = np.pad(np.ones((n, n)), 1)
    return one[:-2, 1:-1] + one[2:, 1:-1] + one[1:-1, :-2] + one[1:-1, 2:]
def unfolded(z):
    """(right, smallest |doubled area| / mean, folded fraction): every triangle of the map turning the same way."""
    z = z - z.reshape(-1, 2).mean(0)
    a, b, c, d = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    cr = lambda o, x, y: (x[..., 0] - o[..., 0]) * (y[..., 1] - o[..., 1]) - (x[..., 1] - o[..., 1]) * (y[..., 0] - o[..., 0])
    t = np.concatenate([cr(a, b, d).ravel(), cr(a, d, c).ravel()])
    s = np.sign(np.median(t))
    if s == 0: return False, 0.0, 1.0
    bad = (t * s <= 0).mean()
    return bad == 0, float(np.abs(t).min() / (np.abs(t).mean() + 1e-300)), float(bad)
def warp(z):
    n = z.shape[0]
    r, c = np.divmod(np.arange(n * n), n)
    a = np.stack([c, r, np.ones(n * n)], 1).astype(float)
    zz = z.reshape(-1, 2); zz = zz - zz.mean(0)
    f, *_ = np.linalg.lstsq(a, zz, rcond=None)
    return float(np.sqrt(((a @ f - zz) ** 2).sum() / ((zz ** 2).sum() + 1e-300)))
