"""Coordinates from two shared numbers: diffuse, remember, then anchor (after an external research brief).

    python coords.py [--size N] [--seeds K] [--extra X]

A square grid of identical cells, each sharing only two numbers, u and v, and seeing only their averages over its side
neighbours (2 at a corner, 3 on an edge, 4 inside). Nothing else: no positions, directions, size, clock or global
statistics. Cells are activated one at a time, at random (with replacement). u and v start as random noise.

Every cell diffuses: it moves (u, v) a little towards its neighbours' averages. From noise, diffusion wipes out
short-scale structure first; what survives longest is the two slowest patterns a square holds (west-east and
south-north), mixed at random, and the rate at which they fade (about pi^2 / n^2) tells the fabric's timescale
without telling its size. Only the four corners (the cells with two neighbours) do more:
  1. A corner watches its proposed change fade: it fits the log of its size against its own age (a leaky,
     age-weighted regression), and grows a confidence while the fade is a clean exponential (fit above 0.98) and
     has run long enough (age x rate above 10).
  2. When confident, it saves its current (u, v), once, and carries on diffusing.
  3. When the rest has faded much further (age since saving x rate above 8), it restores the saved pair and holds
     it for good.
With the four corners held, everyone else averaging settles to a harmonic map between them: unfolded whenever the
four saved points form a convex quadrilateral in the corners' order round the sheet. The coordinates are a warped,
arbitrarily oriented and scaled image of the square, held as tiny differences around the starting mean.

Variants (switches below; see LESSONS.md): BOOST (save the coordinate, change / rate x e^(rate x age), not the pair),
EVERYONE (every cell remembers, not only the corners), SOFT (hold softly: saved pair plus neighbours' average), and
LEAK, SMOOTH, SURE, WAIT. The best so far: EVERYONE, BOOST, SURE 8, WAIT 6, SMOOTH 0.01, LEAK 2, SOFT 0.02.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit

H = 0.001           # step size per activation (and age added per activation)
FIT = 0.98          # how clean an exponential the fade must be
SURE = 10.0         # age x rate before a corner trusts the fade
CONFIDENT = 0.9     # confidence at which a corner saves its pair
EVERYONE = False    # every cell (not only the corners) watches its fade, saves its pair, and holds it
BOOST = False       # save not the pair itself but the cell's coordinate: its proposed change / the fade rate (its offset
                    # from the sheet's average) x e^(rate x age) (undoing the fade): the same scale for every cell,
                    # whenever it saves, and as large as the slow patterns were at the start
SMOOTH = 0.0        # with BOOST: a cell saves a running average of its proposed change (this much new per activation),
                    # not one noisy snapshot (0: the snapshot)
LEAK = 8.0          # the regression's forgetting: weight LEAK x H / age per observation (smaller: a longer memory)
SOFT = 0.0          # once holding: 0 holds the saved pair exactly; above 0, each update moves to this much of the saved
                    # pair plus the rest of the neighbours' average (a sheet pinned everywhere, softly: smooths
                    # cell-to-cell errors, and noise can't add up)
JITTER = 0.0        # ongoing noise added to u and v at each update (the brief's baseline has none)
WAIT = 8.0          # age since saving x rate before a corner holds its saved pair

# a corner's private numbers
AGE, M_A, M_L, C, V_A, V_L, RATE, R2, Q, SAVED, SAVED_U, SAVED_V, SAVED_AGE, SAVED_RATE, HELD, SEEN, GS_U, GS_V = range(18)
PRIVATE = 18


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _activate(u, v, deg, nbr, priv, count, h, fit, sure, confident, wait, jitter, everyone, boost, smooth, leak, soft):
    """count activations of random cells. nbr[i] lists cell i's side neighbours (-1 for none); priv holds each cell's
    private numbers (only corners use more than the age). Returns how many corners hold their pair."""
    n = u.shape[0]
    for _ in range(count):
        i = np.random.randint(n)
        p = priv[i]
        if p[HELD] > 0.5:
            if soft > 0:                                # held softly: its saved pair, plus its neighbours' average
                mu = mv = 0.0
                for k in range(4):
                    j = nbr[i, k]
                    if j >= 0:
                        mu += u[j]
                        mv += v[j]
                u[i] = soft * p[SAVED_U] + (1 - soft) * mu / deg[i] + jitter * np.random.standard_normal()
                v[i] = soft * p[SAVED_V] + (1 - soft) * mv / deg[i] + jitter * np.random.standard_normal()
            continue                                    # held for good: keeps showing its saved pair
        p[AGE] += h
        a = p[AGE]
        mu = mv = 0.0
        for k in range(4):
            j = nbr[i, k]
            if j >= 0:
                mu += u[j]
                mv += v[j]
        d = deg[i]
        gu, gv = d * (u[i] - mu / d), d * (v[i] - mv / d)
        if smooth > 0:
            p[GS_U] += smooth * (gu - p[GS_U])
            p[GS_V] += smooth * (gv - p[GS_V])
        else:
            p[GS_U], p[GS_V] = gu, gv
        if d == 2 or everyone:                          # a corner (or, with EVERYONE, any cell): watch the fade
            ell = 0.5 * np.log(1e-300 + p[GS_U] * p[GS_U] + p[GS_V] * p[GS_V])   # (the running average, if SMOOTH)
            beta = min(1.0, leak * h / a)
            if p[SEEN] < 0.5:
                p[M_A], p[M_L], p[C], p[V_A], p[V_L], p[SEEN] = a, ell, 0.0, 0.0, 0.0, 1.0
            da, dl = a - p[M_A], ell - p[M_L]
            p[M_A] += beta * da
            p[M_L] += beta * dl
            p[C] = (1 - beta) * (p[C] + beta * da * dl)
            p[V_A] = (1 - beta) * (p[V_A] + beta * da * da)
            p[V_L] = (1 - beta) * (p[V_L] + beta * dl * dl)
            if p[V_A] > 1e-12:
                p[RATE] = -p[C] / p[V_A]
            p[R2] = p[C] * p[C] / (1e-300 + p[V_A] * p[V_L])
            good = 1.0 if (a * p[RATE] > sure and p[R2] > fit) else 0.0
            p[Q] += beta * (good - p[Q])
            if p[SAVED] < 0.5 and p[Q] > confident:     # remember the pair, once (before this update)
                p[SAVED], p[SAVED_U], p[SAVED_V], p[SAVED_AGE], p[SAVED_RATE] = 1.0, u[i], v[i], a, p[RATE]
                if boost and p[RATE] > 0:
                    grow = np.exp(min(700.0, p[RATE] * a)) / p[RATE]
                    p[SAVED_U], p[SAVED_V] = -p[GS_U] * grow, -p[GS_V] * grow
            if p[SAVED] > 0.5 and (a - p[SAVED_AGE]) * p[SAVED_RATE] >= wait:
                u[i], v[i], p[HELD] = p[SAVED_U], p[SAVED_V], 1.0      # publish it, and hold it for good
                continue
        u[i] -= h * gu - jitter * np.random.standard_normal()
        v[i] -= h * gv - jitter * np.random.standard_normal()
    held = 0
    for i in range(n):
        if deg[i] == 2 and priv[i, HELD] > 0.5:
            held += 1
    return held


def grid(n: int):
    """The square grid: each cell's side neighbours (-1 for none) and its number of them. Cell (r, c) is r * n + c."""
    nbr = -np.ones((n * n, 4), np.int64)
    for r in range(n):
        for c in range(n):
            i = r * n + c
            for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                if 0 <= r + dr < n and 0 <= c + dc < n:
                    nbr[i, k] = (r + dr) * n + c + dc
    return nbr, (nbr >= 0).sum(1).astype(np.float64)


def life(n: int, seed: int = 0, chunk: int = 0):
    """One life, without end: yields (activations so far, u, v, private numbers, corners held) after each chunk of
    activations (default: 10,000 per cell)."""
    rng = np.random.default_rng(seed)
    u, v = rng.standard_normal(n * n), rng.standard_normal(n * n)
    nbr, deg = grid(n)
    priv = np.zeros((n * n, PRIVATE))
    _seed(seed)
    chunk = chunk or 10_000 * n * n
    done = 0
    while True:
        held = _activate(u, v, deg, nbr, priv, chunk, H, FIT, SURE, CONFIDENT, WAIT, JITTER, EVERYONE, BOOST, SMOOTH, LEAK, SOFT)
        done += chunk
        yield done, u, v, priv, held


# --- the observer: everything below is outside the cells (diagnosis and pictures only) ---

def corners(n: int):
    return [0, n - 1, n * n - 1, n * (n - 1)]            # in order round the sheet


def equilibrium(n: int, priv) -> np.ndarray:
    """The map the cells settle to once the four corners hold: every other cell the average of its side neighbours
    (solved directly)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    nbr, deg = grid(n)
    a = sp.lil_matrix((n * n, n * n))
    bu, bv = np.zeros(n * n), np.zeros(n * n)
    fixed = {i: (priv[i, SAVED_U], priv[i, SAVED_V]) for i in corners(n)}
    for i in range(n * n):
        a[i, i] = 1.0
        if i in fixed:
            bu[i], bv[i] = fixed[i]
            continue
        for j in nbr[i]:
            if j >= 0:
                a[i, j] -= 1.0 / deg[i]
    a = a.tocsr()
    return np.stack([spl.spsolve(a, bu), spl.spsolve(a, bv)], axis=1)


def check(n: int, z: np.ndarray, priv) -> tuple[bool, str]:
    """The brief's test of a map z (n*n x 2): four saved targets; every triangle (each grid square split along the same
    diagonal) turning the same way, with doubled area above 1e-12 after centring and scaling to unit RMS; the edge
    not crossing itself; and the centred map's smaller singular value above 0.001 of the larger."""
    if not all(priv[i, SAVED] > 0.5 for i in corners(n)):
        return False, "not all four corners saved"
    z = z - z.mean(0)
    rms = np.sqrt((z ** 2).sum(1).mean())
    if rms == 0:
        return False, "flat"
    z = z / rms
    s = np.linalg.svd(z, compute_uv=False)
    ratio = s[1] / s[0]
    p = z.reshape(n, n, 2)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    cross = lambda o, x, y: (x[..., 0] - o[..., 0]) * (y[..., 1] - o[..., 1]) - (x[..., 1] - o[..., 1]) * (y[..., 0] - o[..., 0])
    areas = np.concatenate([cross(a, b, d).ravel(), cross(a, d, c).ravel()])
    same_way = (areas > 0).all() or (areas < 0).all()
    small = np.abs(areas).min()
    ring = [p[0, c] for c in range(n)] + [p[r, n - 1] for r in range(1, n)] + \
           [p[n - 1, c] for c in range(n - 2, -1, -1)] + [p[r, 0] for r in range(n - 2, 0, -1)]
    crossings = _crossings(np.array(ring))
    ok = same_way and small > 1e-12 and ratio > 0.001 and crossings == 0
    return ok, (f"triangles {'all one way' if same_way else 'folded'}, smallest doubled area {small:.1e}, "
                f"singular-value ratio {ratio:.3f}, edge crossings {crossings}")


def warp(n: int, z: np.ndarray) -> float:
    """How far the map is from an even grid: the residual of the best affine fit of the map to the cells' true
    positions, as a fraction of the map's spread (0 for an even, straight grid)."""
    r, c = np.divmod(np.arange(n * n), n)
    a = np.stack([c, r, np.ones(n * n)], 1).astype(float)
    z = z - z.mean(0)
    fit, *_ = np.linalg.lstsq(a, z, rcond=None)
    return float(np.sqrt(((a @ fit - z) ** 2).sum() / (z ** 2).sum()))


def _crossings(ring: np.ndarray) -> int:
    """How many pairs of non-adjacent edge segments cross."""
    m = len(ring)
    seg = [(ring[k], ring[(k + 1) % m]) for k in range(m)]
    side = lambda o, x, y: np.sign((x[0] - o[0]) * (y[1] - o[1]) - (x[1] - o[1]) * (y[0] - o[0]))
    count = 0
    for i in range(m):
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue
            (p1, p2), (q1, q2) = seg[i], seg[j]
            if side(p1, p2, q1) * side(p1, p2, q2) < 0 and side(q1, q2, p1) * side(q1, q2, p2) < 0:
                count += 1
    return count


def trial(n: int, seed: int, extra: float = 0.0, give_up: int = 0):
    """A life until all four corners hold (or give_up activations per cell), then optionally extra x that long again
    of continued activations. Returns (held, activations per cell, the equilibrium check, the live map's check after
    the extra time, raw contrast)."""
    give_up = give_up or 400 * n * n * 1000
    for done, u, v, priv, held in life(n, seed):
        if held == 4 or done / (n * n) > give_up:
            break
    per_cell = done / (n * n)
    eq_ok, eq_detail = check(n, equilibrium(n, priv), priv) if held == 4 else (False, "not all held")
    live = None
    if held == 4 and extra > 0:
        target = done * (1 + extra)
        for done2, u, v, priv, held in life_continue(n, u, v, priv, seed, done, target):
            pass
        live = check(n, np.stack([u, v], 1), priv)
    z = np.stack([u, v], 1)
    contrast = np.sqrt(((z - z.mean(0)) ** 2).sum(1).mean())
    return held, per_cell, (eq_ok, eq_detail), live, contrast


def life_continue(n, u, v, priv, seed, start, target):
    nbr, deg = grid(n)
    _seed(seed + 7919)
    chunk = 10_000 * n * n
    done = start
    while done < target:
        held = _activate(u, v, deg, nbr, priv, chunk, H, FIT, SURE, CONFIDENT, WAIT, JITTER, EVERYONE, BOOST, SMOOTH, LEAK, SOFT)
        done += chunk
        yield done, u, v, priv, held


def watch(n: int = 16, seed: int = 0) -> None:
    """A life, live. Left, the sheet: each cell shaded by a checkerboard of eighths of its coordinates (stretched to
    fill the range for display only: the cells' own numbers are tiny differences around a mean), the corners
    lettered by state (. watching, s saved, H held). Right, the map itself: every cell drawn at its (u, v), joined to
    its side neighbours, so a fold shows as crossing lines. Keys: r new seed; space pauses; up and down speed; Escape
    quits."""
    import pygame
    pygame.init()
    side = 480
    scale = side // n
    screen = pygame.display.set_mode((2 * side + 12, side + 50))
    pygame.display.set_caption("coords: two shared numbers, diffuse, remember, then anchor")
    font = pygame.font.SysFont(None, 22)
    per_frame, paused = 1, False
    clock = pygame.time.Clock()
    lives = life(n, seed, chunk=200 * n * n)
    done, u, v, priv, held = next(lives)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    per_frame *= 2
                if event.key == pygame.K_DOWN:
                    per_frame = max(1, per_frame // 2)
                if event.key == pygame.K_r:
                    seed += 1
                    lives = life(n, seed, chunk=200 * n * n)
        if not paused:
            for _ in range(per_frame):
                done, u, v, priv, held = next(lives)
        z = np.stack([u, v], 1)
        z = z - z.mean(0)
        lo, hi = z.min(0), z.max(0)
        w = (z - lo) / np.where(hi - lo > 0, hi - lo, 1)               # display only
        bands = (np.floor(8 * np.clip(w[:, 0], 0, 0.999)) + np.floor(8 * np.clip(w[:, 1], 0, 0.999))) % 2
        shade = (60 + 120 * bands).reshape(n, n)
        screen.fill((0, 0, 0))
        img = np.repeat(np.repeat(shade.astype(np.uint8), scale, 0), scale, 1).T
        screen.blit(pygame.surfarray.make_surface(np.stack([img] * 3, axis=-1)), (0, 0))
        for i in corners(n):
            r, c = divmod(i, n)
            mark = "H" if priv[i, HELD] > 0.5 else ("s" if priv[i, SAVED] > 0.5 else ".")    # (corners only)
            screen.blit(font.render(mark, True, (0, 0, 0), (255, 255, 255)), (min(c * scale, side - 14), min(r * scale, side - 16)))
        pts = side + 12 + 10 + (w * (side - 20)).astype(int)
        pts[:, 1] = 10 + (w[:, 1] * (side - 20)).astype(int)
        p = pts.reshape(n, n, 2)
        for r in range(n):
            for c in range(n):
                if c + 1 < n:
                    pygame.draw.line(screen, (200, 200, 200), p[r, c], p[r, c + 1])
                if r + 1 < n:
                    pygame.draw.line(screen, (200, 200, 200), p[r, c], p[r + 1, c])
        ok, detail = check(n, np.stack([u, v], 1), priv) if held == 4 else (False, f"{held}/4 corners held")
        status = f"seed {seed}   {done / (n * n) / 1e3:7.0f}k activations per cell   {'right' if ok else 'not yet'}: {detail}"
        screen.blit(font.render(status[:150], True, (255, 255, 255)), (6, side + 6))
        screen.blit(font.render("corners: . watching  s saved  H held     r new seed   space   up/down speed   Esc",
                                True, (255, 255, 255)), (6, side + 28))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--jitter", type=float, default=0.0, help="ongoing noise at each update")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--best", action="store_true", help="the best variant so far: everyone remembers, held softly")
    parser.add_argument("--extra", type=float, default=0.0, help="carry on this many times as long again, and check the live map")
    args = parser.parse_args()
    JITTER = args.jitter
    if args.best:
        EVERYONE, BOOST, SURE, WAIT, SMOOTH, LEAK, SOFT = True, True, 8.0, 6.0, 0.01, 2.0, 0.02
    if args.watch:
        watch(args.size)
        raise SystemExit
    for seed in range(args.seeds):
        t0 = time.time()
        held, per_cell, (ok, detail), live, contrast = trial(args.size, seed, args.extra)
        extra = f"; live map after {args.extra:g}x more: {'right' if live[0] else 'WRONG'} ({live[1]})" if live else ""
        print(f"{args.size}x{args.size} seed {seed}: {held}/4 corners held after {per_cell / 1e6:.2f}M activations per cell; "
              f"equilibrium {'right' if ok else 'WRONG'} ({detail}); contrast {contrast:.1e}{extra}  ({time.time() - t0:.0f} s)", flush=True)
