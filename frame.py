"""A body frame: on a square sheet of identical cells (all starting the same, bar noise), every cell comes to hold two
coordinates, x and y from 0 to 1, with no global knowledge; then cells name themselves as particular places by those
coordinates (here the middle of the north edge, and the point halfway from the centre to the north-east corner).

    python frame.py [--size N] [--seed N] [--dt X] [--time T] [--relax W] [--clean] [--picture FILE.png] [--watch]

In sequence, each step simple:
  1. Corners know they are corners: they have only three neighbours.
  2. Election: every cell holds the smallest corner tag it has heard (each cell is born with a random tag, as a
     name). The corner whose own tag survives is the origin, x = y = 0. It waits SETTLE before acting, so that early
     false winners (each corner believes it wins until a smaller tag arrives) never act. No ties, one crossing.
  3. The origin chooses the axes itself: of its two neighbours along the edges, the one with the smaller tag starts a
     sign +1 and the other -1. Signs run along the edge cells (no counting; an edge cell keeps the first it gets).
     The corner +1 reaches is x = 1, y = 0, and relays +2 along its other edge; the corner +2 reaches is x = y = 1;
     the corner -1 reaches is x = 0, y = 1. No corner ever sees a mixture of signs, so there is no race.
  4. Coordinates: labelled corners hold their 0s and 1s, and every other cell averages its neighbours' x and y (the
     steady state of diffusion). With only the corners held, the coordinates are warped (most of the sheet sits near
     0.5, and they change steeply near the corners), but they are smooth and unique, and a place is named by its
     warped coordinates. RELAX above 1 over-relaxes (each update overshoots the average a little), which settles far
     faster.
  5. Naming a place: once a cell knows its coordinates, it takes a name if its (x, y) is nearer the place's
     coordinates than any neighbour's. A cell knows its coordinates when its counts from opposite edges add up to the
     same as each neighbour's: in a finished frame those sums are the same everywhere (the width and the height), and
     while the edges are still being labelled they aren't. (Without that check, cells name themselves from their
     starting noise, or from counts that came from only part of an edge: thousands of false claims on a big sheet.)

Every cell updates at random times (each tick with chance UPDATE), and its x and y get random jitter (JITTER; --clean
turns it off). A cell's numbers: the smallest tag heard, how long it has been the winner (corners), its sign (edges and
corners), and x and y.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit, prange

SIZE = 49
UPDATE = 0.5        # the chance that a cell updates in a tick
JITTER = 0.001      # random jitter in x and y at each update
SETTLE = 300.0      # how long a corner must have been the winner before it acts as the origin
RULE = "waves"      # waves: counts in from each edge, coordinates as fractions; average: neighbours' average
RELAX = 1.0         # how far towards (or past) its neighbours' average a cell moves its x and y per update
NOISE = 0.1         # cells start with x and y uniform in 0..NOISE
TIE = 1e-4          # how much nearer a neighbour must be to a place before a cell gives up its name
PLACES = {"N": (0.5, 1.0), "H": (0.75, 0.75)}   # named places, as true fractions of the body: north middle, NE halfway

# The cells' numbers (MIN: smallest corner tag heard; LEAD: time a corner has been the winner; SIGN: 0 none, +1, -1,
# +2; X, Y) and what each cell has decided (CORNER: 0 none, 1 origin, 2 (1,0), 3 (0,1), 4 (1,1); NAME: 0 none, else the
# place's number).
MIN, LEAD, SIGN, CORNER, X, Y, NAME, D_S, D_W, D_E, D_N = range(11)   # D_*: counts in from each edge (waves)
LAYERS = 11
FAR = 1e6
CORNER_XY = ((-1.0, -1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


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


@njit(cache=True, parallel=True)
def _ticks(s, s2, tag, kind, targets, ticks, dt, k):
    update, jitter, settle, relax, tie, waves = k[0], k[1], k[2], k[3], k[4], k[5] > 0.5
    n = s.shape[1]
    for _ in range(ticks):
        for y in prange(n):                                 # the corners' clocks: time as the winner
            for x in range(n):
                if kind[y, x] == 3 and s[MIN, y, x] == tag[y, x]:
                    s[LEAD, y, x] += dt
                else:
                    s[LEAD, y, x] = 0.0
        s2[:, :, :] = s
        for y in prange(n):                                 # rows in parallel: each cell reads only the old state
            for x in range(n):
                if np.random.random() >= update:
                    continue
                corner, edge = kind[y, x] == 3, kind[y, x] == 5
                least = tag[y, x] if corner else 2.0         # the election: the smallest corner tag heard
                total_x = total_y = 0.0
                seen = np.zeros(5)                          # signs seen on orthogonal edge or corner neighbours:
                least_d = np.full(4, FAR)                   # +1, -1, +2, -2 (index 1..4); smallest counts (waves)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = y + dy, x + dx
                        if not ((dy or dx) and 0 <= yy < n and 0 <= xx < n):
                            continue
                        least = min(least, s[MIN, yy, xx])
                        total_x += s[X, yy, xx]
                        total_y += s[Y, yy, xx]
                        for e in range(4):                  # whole-number counts, as in places.py
                            least_d[e] = min(least_d[e], np.round(s[D_S + e, yy, xx]) + 1.0)
                        if (dy == 0 or dx == 0) and kind[yy, xx] != 8:
                            sg = s[SIGN, yy, xx]
                            if sg == 1.0:
                                seen[1] = 1.0
                            elif sg == -1.0:
                                seen[2] = 1.0
                            elif sg == 2.0:
                                seen[3] = 1.0
                            elif sg == -2.0:
                                seen[4] = 1.0
                            # beside the settled origin: of its two edge neighbours (diagonal to each other), the one
                            # with the smaller tag takes +1
                            if (edge and s2[SIGN, y, x] == 0.0 and s[CORNER, yy, xx] == 1.0):
                                for ey in range(-1, 2):
                                    for ex in range(-1, 2):
                                        oy, ox = y + ey, x + ex
                                        if ey and ex and 0 <= oy < n and 0 <= ox < n and kind[oy, ox] == 5 \
                                                and abs(oy - yy) + abs(ox - xx) == 1:
                                            s2[SIGN, y, x] = 1.0 if tag[y, x] < tag[oy, ox] else -1.0
                s2[MIN, y, x] = least
                if edge and s2[SIGN, y, x] == 0.0:           # signs run along the edges: keep the first one got
                    if seen[1] > 0.0:
                        s2[SIGN, y, x] = 1.0
                    elif seen[2] > 0.0:
                        s2[SIGN, y, x] = -1.0
                    elif seen[3] > 0.0:
                        s2[SIGN, y, x] = 2.0
                    elif seen[4] > 0.0:
                        s2[SIGN, y, x] = -2.0
                if corner:                                   # what this corner is
                    if s[MIN, y, x] == tag[y, x] and s[LEAD, y, x] > settle:
                        s2[CORNER, y, x] = 1.0              # the origin
                    elif seen[1] > 0.0:
                        s2[CORNER, y, x], s2[SIGN, y, x] = 2.0, 2.0     # x = 1, y = 0: relays +2
                    elif seen[2] > 0.0:
                        s2[CORNER, y, x], s2[SIGN, y, x] = 3.0, -2.0    # x = 0, y = 1: relays -2
                    elif seen[3] > 0.0 or seen[4] > 0.0:
                        s2[CORNER, y, x] = 4.0              # x = y = 1
                label = int(s2[CORNER, y, x])
                if waves:
                    # which edges this cell is on: south (+1), west (-1), east (+2), north (-2); corners are on two
                    sg = s2[SIGN, y, x]
                    on = (sg == 1.0 or label == 1 or label == 2, sg == -1.0 or label == 1 or label == 3,
                          sg == 2.0 or label == 2 or label == 4, sg == -2.0 or label == 3 or label == 4)
                    for e in range(4):                      # count in from each edge
                        s2[D_S + e, y, x] = (0.0 if on[e] else least_d[e]) + jitter * np.random.standard_normal()
                    ds, dw, de, dn = s2[D_S, y, x], s2[D_W, y, x], s2[D_E, y, x], s2[D_N, y, x]
                    if dw + de < FAR / 2 and ds + dn < FAR / 2 and dw + de > 0.5 and ds + dn > 0.5:
                        s2[X, y, x], s2[Y, y, x] = dw / (dw + de), ds / (ds + dn)    # fractions of the body
                elif label > 0:                              # a labelled corner holds its coordinates
                    s2[X, y, x], s2[Y, y, x] = CORNER_XY[label]
                else:                                        # everyone else averages (over-relaxed by RELAX)
                    m = kind[y, x]
                    s2[X, y, x] = s[X, y, x] + relax * (total_x / m - s[X, y, x]) + jitter * np.random.standard_normal()
                    s2[Y, y, x] = s[Y, y, x] + relax * (total_y / m - s[Y, y, x]) + jitter * np.random.standard_normal()
                # naming: nearer a place's coordinates than any neighbour, once this cell knows its coordinates
                name = 0.0
                # a cell knows its coordinates when its counts from opposite edges add up to the same as its
                # neighbours' (in a finished frame the sums are the same everywhere: the width and the height)
                knows = True
                if waves:
                    across = np.round(s[D_W, y, x]) + np.round(s[D_E, y, x])
                    down = np.round(s[D_S, y, x]) + np.round(s[D_N, y, x])
                    knows = across < FAR / 2 and down < FAR / 2
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            yy, xx = y + dy, x + dx
                            if knows and (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                                knows = (np.round(s[D_W, yy, xx]) + np.round(s[D_E, yy, xx]) == across
                                         and np.round(s[D_S, yy, xx]) + np.round(s[D_N, yy, xx]) == down)
                for p in range(targets.shape[0] if knows else 0):
                    mine = abs(s[X, y, x] - targets[p, 0]) + abs(s[Y, y, x] - targets[p, 1])
                    nearest = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            yy, xx = y + dy, x + dx
                            if (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                                if abs(s[X, yy, xx] - targets[p, 0]) + abs(s[Y, yy, xx] - targets[p, 1]) < mine - tie:
                                    nearest = False
                    if nearest:
                        name = p + 1.0
                s2[NAME, y, x] = name
        s[:, :, :] = s2


def warped(n: int = None):
    """The coordinates averaging settles to, with only the corners held: x (columns) and y (rows, y = 0 at row 0),
    solved directly. Used to give each named place its warped coordinates, as a rule's constants would be."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    n = n or SIZE
    held = {(0, 0): (0, 0), (0, n - 1): (1, 0), (n - 1, 0): (0, 1), (n - 1, n - 1): (1, 1)}   # (row, col): (x, y)
    a = sp.lil_matrix((n * n, n * n))
    bx, by = np.zeros(n * n), np.zeros(n * n)
    for r in range(n):
        for c in range(n):
            i = r * n + c
            a[i, i] = 1.0
            if (r, c) in held:
                bx[i], by[i] = held[(r, c)]
                continue
            nb = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and 0 <= r + dr < n and 0 <= c + dc < n]
            for q in nb:
                a[i, q[0] * n + q[1]] -= 1.0 / len(nb)
    a = a.tocsr()
    return spl.spsolve(a, bx).reshape(n, n), spl.spsolve(a, by).reshape(n, n)


def ideal(n: int = None):
    """The coordinates the rule should settle to: true fractions (waves), or the warped ones (averaging)."""
    n = n or SIZE
    if RULE == "waves":
        cols = np.arange(n) / (n - 1)
        return np.tile(cols, (n, 1)), np.tile(cols[:, None], (1, n))
    if getattr(ideal, "cache", (None,))[0] != n:
        ideal.cache = (n, warped(n))
    return ideal.cache[1]


def places(n: int = None):
    """Each named place: its true cell in frame coordinates (row = y fraction, column = x fraction) and the
    coordinates a cell there should hold."""
    n = n or SIZE
    wx, wy = ideal(n)
    out = {}
    for name, (fx, fy) in PLACES.items():
        r, c = int(round(fy * (n - 1))), int(round(fx * (n - 1)))
        out[name] = ((r, c), (wx[r, c], wy[r, c]))
    return out


def knobs():
    return np.array([UPDATE, JITTER, SETTLE, RELAX, TIE, 1.0 if RULE == "waves" else 0.0])


def life(seed: int = 0, dt: float = 1.0):
    """One life, without end: yields (time, the cells' numbers, tag) after every tick."""
    n = SIZE
    rng = np.random.default_rng(seed)
    s = np.zeros((LAYERS, n, n))
    s[MIN] = 2.0                                            # no corner tag heard yet
    s[D_S:D_N + 1] = FAR                                    # no edge heard from yet
    s[X], s[Y] = rng.uniform(0, NOISE, (n, n)), rng.uniform(0, NOISE, (n, n))
    tag = rng.random((n, n))
    kind = np.array([[_neighbours(n, y, x) for x in range(n)] for y in range(n)])
    targets = np.array([w for _, w in places(n).values()])
    s2 = np.empty_like(s)
    k = knobs()
    _seed(seed)
    step = 0
    while True:
        _ticks(s, s2, tag, kind, targets, 1, dt, k)
        step += 1
        yield step * dt, s, tag


def frame_of(s, tag):
    """Which grid corner each labelled corner is, as a map from frame (row = y, col = x, in cells) to grid cells, or
    None while the corners aren't all labelled."""
    n = s.shape[1]
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    label = {}
    for c in corners:
        code = int(s[CORNER][c])
        if code > 0:
            label[tuple(int(v) for v in CORNER_XY[code])] = c
    if len(label) < 4:
        return None
    o, ex, ey = (np.array(label[k]) for k in ((0, 0), (1, 0), (0, 1)))
    ux, uy = (ex - o) / (n - 1), (ey - o) / (n - 1)          # grid steps per frame step along x and along y
    return lambda r, c: tuple(int(v) for v in np.round(o + ux * c + uy * r))


def judge(s, tag):
    """(right, detail): right if the corners are all labelled consistently and every named place is claimed by
    exactly its own cell; and the largest error in any cell's coordinates against the warped ideal."""
    to_grid = frame_of(s, tag)
    if to_grid is None:
        return False, "corners not all labelled"
    n = s.shape[1]
    wx, wy = ideal(n)
    ok = True
    for i, (name, ((r, c), _)) in enumerate(places(n).items()):
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
    """In greys: the warped coordinate grid as a checkerboard of bands (eighths of x and y), dim; named places white."""
    bands = (np.floor(8 * np.clip(s[X], 0, 0.999)) + np.floor(8 * np.clip(s[Y], 0, 0.999))) % 2
    shade = 0.15 + 0.25 * bands + 0.1 * s[X] * s[Y]
    return np.where(s[NAME] > 0, 1.0, shade)


def save_picture(pictures, path: str, scale: int = 4) -> None:
    import pygame
    strip = np.concatenate([np.pad(p, ((0, 0), (0, 2)), constant_values=0.5) for p in pictures], axis=1)
    grey = (255 * strip).astype(np.uint8).repeat(scale, 0).repeat(scale, 1)
    pygame.image.save(pygame.surfarray.make_surface(np.stack([grey.T] * 3, axis=-1)), path)


def watch(seed: int = 0, dt: float = 1.0, scale: int = 0, title: str = "frame") -> None:
    """A life, live: the warped coordinate grid as a checkerboard of bands (eighths of x and y), the corners lettered
    with their coordinates, and the cells that name themselves white and lettered (N north middle, H NE halfway).
    Keys: r restarts with a new seed; space pauses; up and down arrows change how many ticks pass per frame; Escape
    quits."""
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
        for c in ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)):
            xy = CORNER_XY[int(s[CORNER][c])]
            if xy[0] >= 0:
                label = font.render(f"{int(xy[0])}{int(xy[1])}", True, (255, 255, 255), (60, 60, 60))
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
    parser.add_argument("--rule", choices=("waves", "average"), default=RULE)
    parser.add_argument("--relax", type=float, default=RELAX)
    parser.add_argument("--settle", type=float, default=SETTLE, help="how long the origin waits (more than a crossing)")
    parser.add_argument("--clean", action="store_true", help="no jitter")
    parser.add_argument("--picture", help="save the development as a PNG")
    parser.add_argument("--watch", action="store_true", help="watch it live instead")
    args = parser.parse_args()
    SIZE, RELAX, RULE, SETTLE = args.size, args.relax, args.rule, args.settle
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
