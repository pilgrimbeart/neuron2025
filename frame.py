"""A body frame: on a square sheet of identical cells (all starting the same, bar noise), every cell comes to hold two
coordinates, x and y from 0 to 1, with no global knowledge; then cells name themselves as particular places by them
(here the middle of the north edge, and the point halfway from the centre to the north-east corner).

    python frame.py [--size N] [--seed N] [--dt X] [--time T] [--clean] [--picture FILE.png] [--watch]

A cell shares five numbers with its neighbours: the smallest corner tag it has heard, and four whole-number counts,
in from the south, west, east and north edges. Everything else it keeps to itself (x, y and its name, worked out from
its own counts). In sequence, each step simple:
  1. Corners know they are corners: they have only three neighbours.
  2. Election: every cell holds the smallest corner tag it has heard (each cell is born with a random tag, as a
     name). The corner whose own tag survives is the origin, x = y = 0. Every corner believes it wins until a smaller
     tag reaches it, and acts on that at once; so a cell's counts are kept only while the smallest tag it has heard
     stays the same (when a smaller one arrives it drops them and starts again), and it uses a neighbour's counts
     only if that neighbour has heard the same smallest tag. A false origin's work is discarded as the news of the
     true winner spreads, with no waiting, so it works at any size.
  3. Edges are labelled by zeros in the counts: a cell on the south edge holds 0 as its count from the south, and so
     on. The origin holds 0 from the south and the west. Of its two neighbours along the edges, the one with the
     smaller tag holds 0 from the south, the other 0 from the west. An edge cell takes the zero of an edge neighbour
     (not a corner) along the edge. A corner reached by a zero from the south is (1, 0), and also holds 0 from the
     east; one reached from the west is (0, 1), and holds 0 from the north; one reached from the east or the north
     is (1, 1), and holds both. The edge cell next to a corner takes the zero the corner holds that its other edge
     neighbour (diagonal to it) lacks, if it has no zero yet: the corner's new edge. Whole numbers make the zeros robust
     to noise.
  4. Counts: every other count is the smallest neighbour's (with the same smallest tag) plus one: rows or columns in
     from that edge. Coordinates are fractions: x = from west / (from west + from east), y likewise.
  5. Naming a place: a cell knows its coordinates when its counts from opposite edges add up to the same as each
     neighbour's (in a finished frame those sums are the same everywhere: the width and the height); it then takes a
     name if its (x, y) is nearer the place's than any neighbour's (on a tie, the one to the west, or south, wins). A
     place is a fraction of the body, plus optionally an offset in cells: a cell is 1 / width across, which every
     cell knows from its own counts, so "one cell east of the north middle" is exact at any size.
Every cell updates at random times (each tick with chance UPDATE), and its counts get random jitter (JITTER; --clean
turns it off).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit, prange

SIZE = 49
UPDATE = 0.5        # the chance that a cell updates in a tick
JITTER = 0.001      # random jitter in the counts at each update
TIE = 1e-4          # how much nearer a neighbour must be to a place before a cell gives up its name
PLACES = {"N": (0.5, 1.0), "H": (0.75, 0.75)}   # named places, as fractions of the body: north middle, NE halfway;
                                                # optionally also an offset in cells: (x, y, cells east, cells north)

# Shared with neighbours: MIN (the smallest corner tag heard) and the counts in from each edge (D_S, D_W, D_E, D_N).
# Kept to itself: X, Y (from its counts) and NAME (0 none, else the place's number).
MIN, D_S, D_W, D_E, D_N, X, Y, NAME = range(8)
LAYERS = 8
FAR = 1e6
S, W, E, N = range(4)


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _neighbours(n, y, x):
    """How many neighbours the cell at (y, x) has: 3 at a corner, 5 on an edge, 8 inside."""
    k = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if (dy or dx) and 0 <= y + dy < n and 0 <= x + dx < n:
                k += 1
    return k


@njit(cache=True)
def _zero(v):
    return abs(v) < 0.5


@njit(cache=True, parallel=True)
def _ticks(s, s2, tag, kind, targets, ticks, dt, k):
    update, jitter, tie = k[0], k[1], k[2]
    n = s.shape[1]
    for _ in range(ticks):
        s2[:, :, :] = s
        for y in prange(n):                                 # rows in parallel: each cell reads only the old state
            for x in range(n):
                if np.random.random() >= update:
                    continue
                corner, edge = kind[y, x] == 3, kind[y, x] == 5
                least = tag[y, x] if corner else 2.0         # the election: the smallest corner tag heard
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = y + dy, x + dx
                        if (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                            least = min(least, s[MIN, yy, xx])
                s2[MIN, y, x] = least
                fresh = s[MIN, y, x] != least                # a new winner believed in: start again under it
                zero = np.zeros(4)                          # which edges this cell is on (counts that are 0)
                if not fresh:
                    for e in range(4):
                        zero[e] = 1.0 if _zero(s[D_S + e, y, x]) else 0.0
                labelled = zero.sum()
                near = np.full(4, FAR)                      # the smallest neighbour's count + 1, under the same winner
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = y + dy, x + dx
                        if not ((dy or dx) and 0 <= yy < n and 0 <= xx < n) or s[MIN, yy, xx] != least:
                            continue
                        for e in range(4):
                            near[e] = min(near[e], np.round(s[D_S + e, yy, xx]) + 1.0)
                        if not (edge or corner) or (dy and dx) or kind[yy, xx] == 8:
                            continue                        # below: edge cells and corners, and their neighbours
                                                            # along the edge
                        if kind[yy, xx] == 5:               # an edge neighbour's zero runs on along the edge
                            for e in range(4):
                                if _zero(s[D_S + e, yy, xx]):
                                    zero[e] = 1.0
                        elif tag[yy, xx] == least:          # beside the origin: the smaller tag of its two edge
                                                            # neighbours takes the south edge, the other the west
                            for ey in range(-1, 2):
                                for ex in range(-1, 2):
                                    oy, ox = y + ey, x + ex
                                    if ey and ex and 0 <= oy < n and 0 <= ox < n and kind[oy, ox] == 5 \
                                            and abs(oy - yy) + abs(ox - xx) == 1:
                                        zero[S if tag[y, x] < tag[oy, ox] else W] = 1.0
                        elif labelled == 0.0:               # beside another corner, with no zero of my own yet:
                            for e in range(4):              # take its zeros that my other edge neighbour lacks
                                if _zero(s[D_S + e, yy, xx]):
                                    lacks = True
                                    for ey in range(-1, 2):
                                        for ex in range(-1, 2):
                                            oy, ox = y + ey, x + ex
                                            if ey and ex and 0 <= oy < n and 0 <= ox < n and kind[oy, ox] == 5 \
                                                    and abs(oy - yy) + abs(ox - xx) == 1 \
                                                    and s[MIN, oy, ox] == least and _zero(s[D_S + e, oy, ox]):
                                                lacks = False
                                    if lacks:
                                        zero[e] = 1.0
                if corner:                                  # what this corner is
                    if least == tag[y, x]:
                        zero[S] = zero[W] = 1.0             # the origin
                    elif zero[S] > 0:
                        zero[E] = 1.0                       # (1, 0)
                    elif zero[W] > 0:
                        zero[N] = 1.0                       # (0, 1)
                    elif zero[E] > 0 or zero[N] > 0:
                        zero[E] = zero[N] = 1.0             # (1, 1)
                for e in range(4):                          # count in from each edge
                    s2[D_S + e, y, x] = (0.0 if zero[e] > 0 else near[e]) + jitter * np.random.standard_normal()
                ds, dw = np.round(s2[D_S, y, x]), np.round(s2[D_W, y, x])
                de, dn = np.round(s2[D_E, y, x]), np.round(s2[D_N, y, x])
                if dw + de < FAR / 2 and ds + dn < FAR / 2 and dw + de > 0.5 and ds + dn > 0.5:
                    s2[X, y, x], s2[Y, y, x] = dw / (dw + de), ds / (ds + dn)    # fractions of the body
                # naming: once this cell knows its coordinates (its sums of opposite counts equal its neighbours'),
                # nearer a place's coordinates than any neighbour (whose coordinates it works out from their counts)
                across = np.round(s[D_W, y, x]) + np.round(s[D_E, y, x])
                down = np.round(s[D_S, y, x]) + np.round(s[D_N, y, x])
                knows = across < FAR / 2 and down < FAR / 2 and across > 0.5 and down > 0.5 and not fresh
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = y + dy, x + dx
                        if knows and (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                            knows = (s[MIN, yy, xx] == least
                                     and np.round(s[D_W, yy, xx]) + np.round(s[D_E, yy, xx]) == across
                                     and np.round(s[D_S, yy, xx]) + np.round(s[D_N, yy, xx]) == down)
                name = 0.0
                for p in range(targets.shape[0] if knows else 0):
                    tx = targets[p, 0] + targets[p, 2] / across     # the place: a fraction of the body, plus any
                    ty = targets[p, 1] + targets[p, 3] / down       # offset in cells (a cell is 1 / width across)
                    mx, my = np.round(s[D_W, y, x]) / across, np.round(s[D_S, y, x]) / down
                    mine = abs(mx - tx) + abs(my - ty)
                    nearest = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            yy, xx = y + dy, x + dx
                            if (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                                ox, oy = np.round(s[D_W, yy, xx]) / across, np.round(s[D_S, yy, xx]) / down
                                theirs = abs(ox - tx) + abs(oy - ty)
                                if theirs < mine - tie or (theirs <= mine + tie and (ox < mx or (ox == mx and oy < my))):
                                    nearest = False         # nearer, or as near and to the west (or south)
                    if nearest:
                        name = p + 1.0
                s2[NAME, y, x] = name
        s[:, :, :] = s2


def ideal(n: int = None):
    """The coordinates the rule should settle to, in frame coordinates (row = y, column = x): true fractions."""
    n = n or SIZE
    cols = np.arange(n) / (n - 1)
    return np.tile(cols, (n, 1)), np.tile(cols[:, None], (1, n))


def places(n: int = None):
    """Each named place: the cell it should be, in frame coordinates (row = y, column = x), for judging."""
    n = n or SIZE
    out = {}
    for name, p in PLACES.items():
        fx, fy, ox, oy = (tuple(p) + (0, 0))[:4]
        nearest = lambda v: int(np.ceil(v - 0.5 - 1e-9))     # ties go west (or south), as in the cells
        out[name] = (nearest(fy * (n - 1)) + oy, nearest(fx * (n - 1)) + ox)
    return out


def targets_of():
    """The places as the cells get them: fractions of the body, and offsets in cells (no size in them)."""
    return np.array([(tuple(p) + (0, 0))[:4] for p in PLACES.values()], dtype=float)


def knobs():
    return np.array([UPDATE, JITTER, TIE])


def life(seed: int = 0, dt: float = 1.0):
    """One life, without end: yields (time, the cells' numbers, tag) after every tick."""
    n = SIZE
    rng = np.random.default_rng(seed)
    s = np.zeros((LAYERS, n, n))
    s[MIN] = 2.0                                            # no corner tag heard yet
    s[D_S:D_N + 1] = FAR                                    # no edge heard from yet
    s[X], s[Y] = rng.uniform(0, 0.1, (n, n)), rng.uniform(0, 0.1, (n, n))
    tag = rng.random((n, n))
    kind = np.array([[_neighbours(n, y, x) for x in range(n)] for y in range(n)])
    targets = targets_of()
    s2 = np.empty_like(s)
    k = knobs()
    _seed(seed)
    step = 0
    while True:
        _ticks(s, s2, tag, kind, targets, 1, dt, k)
        step += 1
        yield step * dt, s, tag


def corners_of(s):
    """Which grid corner is (0, 0), (1, 0), (0, 1) and (1, 1), by the zeros in its counts, or None while they aren't
    all labelled consistently."""
    n = s.shape[1]
    zero = lambda c, e: abs(s[D_S + e][c]) < 0.5
    label = {}
    for c in ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)):
        on = tuple(e for e in range(4) if zero(c, e))
        xy = {(S, W): (0, 0), (S, E): (1, 0), (W, N): (0, 1), (E, N): (1, 1)}.get(on)
        if xy is None or xy in label:
            return None
        label[xy] = c
    return label


def frame_of(s, tag):
    """A map from frame (row = y, col = x, in cells) to grid cells, or None while the corners aren't all labelled."""
    label = corners_of(s)
    if label is None:
        return None
    n = s.shape[1]
    o, ex, ey = (np.array(label[k]) for k in ((0, 0), (1, 0), (0, 1)))
    ux, uy = (ex - o) / (n - 1), (ey - o) / (n - 1)          # grid steps per frame step along x and along y
    return lambda r, c: tuple(int(v) for v in np.round(o + ux * c + uy * r))


def judge(s, tag):
    """(right, detail): right if the corners are all labelled consistently and every named place is claimed by
    exactly its own cell; and the largest error in any cell's coordinates."""
    to_grid = frame_of(s, tag)
    if to_grid is None:
        return False, "corners not all labelled"
    n = s.shape[1]
    wx, wy = ideal(n)
    ok = True
    for i, (name, (r, c)) in enumerate(places(n).items()):
        claimed = {tuple(int(v) for v in q) for q in np.argwhere(s[NAME] == i + 1)}
        ok = ok and claimed == {to_grid(r, c)}
    err = max(abs(s[X][to_grid(r, c)] - wx[r, c]) + abs(s[Y][to_grid(r, c)] - wy[r, c])
              for r in range(0, n, 4) for c in range(0, n, 4))
    return ok, f"coordinate error up to {err:.3f}"


def run(seed=0, dt=1.0, time=20000.0, every=50.0):
    """When the named places are first right, and how much of the time after that they stay right."""
    first, right, checks, detail = None, 0, 0, ""
    for t, s, tag in life(seed, dt):
        if abs(t / every - round(t / every)) < 1e-9:
            ok, detail = judge(s, tag)
            if first is None and ok:
                first = t
            if first is not None:
                checks += 1
                right += ok
        if t >= time - dt / 2:
            return first, (right / checks if checks else 0.0), detail


def picture(s, tag) -> np.ndarray:
    """In greys: the coordinate grid as a checkerboard of bands (eighths of x and y), dim; named places white."""
    bands = (np.floor(8 * np.clip(s[X], 0, 0.999)) + np.floor(8 * np.clip(s[Y], 0, 0.999))) % 2
    shade = 0.15 + 0.25 * bands + 0.1 * s[X] * s[Y]
    return np.where(s[NAME] > 0, 1.0, shade)


def save_picture(pictures, path: str, scale: int = 4) -> None:
    import pygame
    strip = np.concatenate([np.pad(p, ((0, 0), (0, 2)), constant_values=0.5) for p in pictures], axis=1)
    grey = (255 * strip).astype(np.uint8).repeat(scale, 0).repeat(scale, 1)
    pygame.image.save(pygame.surfarray.make_surface(np.stack([grey.T] * 3, axis=-1)), path)


def watch(seed: int = 0, dt: float = 1.0, scale: int = 0, title: str = "frame") -> None:
    """A life, live: the coordinate grid as a checkerboard of bands (eighths of x and y), the corners lettered with
    their coordinates, and the cells that name themselves white and lettered (N north middle, H NE halfway). Keys: r
    restarts with a new seed; space pauses; up and down arrows change how many ticks pass per frame; Escape quits."""
    import pygame
    pygame.init()
    n = SIZE
    scale = scale or max(1, 640 // n)
    side = n * scale
    screen = pygame.display.set_mode((side, side + 50))
    pygame.display.set_caption(title)
    font = pygame.font.SysFont(None, 20)
    ticks_per_frame, paused = max(4, n // 32), False
    clock = pygame.time.Clock()
    lives = life(seed, dt)
    names = list(PLACES)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    ticks_per_frame *= 2
                if event.key == pygame.K_DOWN:
                    ticks_per_frame = max(1, ticks_per_frame // 2)
                if event.key == pygame.K_r:
                    seed += 1
                    lives = life(seed, dt)
        if paused:
            clock.tick(30)
            continue
        started = time.time()                       # as many ticks as fit in a fifth of a second, so it never stalls
        done = 0
        while done < ticks_per_frame and (done == 0 or time.time() - started < 0.2):
            t, s, tag = next(lives)
            done += 1
        rate = done / max(time.time() - started, 1e-6)
        grey = (255 * picture(s, tag)).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        for xy, c in (corners_of(s) or {}).items():
            label = font.render(f"{xy[0]}{xy[1]}", True, (255, 255, 255), (60, 60, 60))
            screen.blit(label, (min(c[1] * scale, side - 18), min(c[0] * scale, side - 14)))
        for r, c in np.argwhere(s[NAME] > 0)[:20]:                  # (only a few letters: drawing is slow)
            screen.blit(font.render(names[int(s[NAME][r, c]) - 1], True, (0, 0, 0)), (c * scale + 2, r * scale))
        ok, detail = judge(s, tag)
        status = f"seed {seed}   time {t:7.0f}   {'right' if ok else 'not yet'}   {detail}   {rate:5.0f} ticks/s (up to {ticks_per_frame}/frame)"
        screen.blit(font.render(status, True, (255, 255, 255)), (6, side + 6))
        screen.blit(font.render("r new seed   space pause   up/down speed   Esc quit", True, (255, 255, 255)),
                    (6, side + 28))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=20000.0)
    parser.add_argument("--clean", action="store_true", help="no jitter")
    parser.add_argument("--picture", help="save the development as a PNG")
    parser.add_argument("--watch", action="store_true", help="watch it live instead")
    args = parser.parse_args()
    SIZE = args.size
    if args.clean:
        JITTER = 0.0
    if args.watch:
        watch(args.seed, args.dt)
        raise SystemExit
    first, stays, detail = run(args.seed, args.dt, args.time)
    print(f"named places right first at {first}, right {100 * stays:.0f}% of the time after; {detail}")
    if args.picture:
        pics, marks = [], [args.time * f for f in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)]
        for t, s, tag in life(args.seed, args.dt):
            if marks and t >= marks[0] - args.dt / 2:
                pics.append(picture(s, tag)); marks.pop(0)
            if not marks:
                break
        save_picture(pics, args.picture)
