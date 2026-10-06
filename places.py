"""Cells that know where they are: on a square sheet of identical cells (all starting the same, bar noise), the four
corner cells and the centre cells come to identify themselves, with no global knowledge.

    python places.py [--rule steps|leak] [--size N] [--seed N] [--dt X] [--time T] [--clean] [--picture FILE.png]
                     [--watch]

In a uniform sheet the only thing that differs from place to place is the edge, so all positional information has to
come from there. A cell senses the edge locally: beyond it, it simply has no neighbour. Two rules:

  steps  Count steps in from the edge, then out from the centre. A cell at the edge holds 0; every other cell moves
         its count towards the smallest of its neighbours' plus one. The centre is where that count is highest: no
         neighbour's is higher. Counts are read as whole numbers (rounded), so that noise under half a step is wiped
         out at every step instead of piling up along the count (as digital signals resist noise). The centre then holds 0 in a
         second count, moving only along rows and columns (a diagonal step counts 2, as a rook moves), and the
         corners are where that count is highest: on a square, the corners are the points farthest from the centre,
         and counted this way the count climbs a whole step per cell along each edge towards them (with diagonal
         steps of sqrt 2 it climbs only 0.41, too little above the noise).
  leak   A leaky chemical, in one variable. Every cell makes a little (MAKE per unit time); it spreads by averaging
         with the eight neighbour slots, and a slot beyond the edge holds none, so it leaks away there. It settles
         highest at the centre and lowest at the corners (which leak on five sides): the centre is a cell no neighbour
         exceeds, a corner one clearly below all its neighbours (the edge middles are flat, so merely "no neighbour lower"
         would let noise make false corners there).

Every cell updates at random times (each tick with chance UPDATE), and each value it computes gets random jitter
(JITTER; --clean turns it off). A cell decides it is the centre (or a corner) only by comparing its own value with its
neighbours' largest (or smallest), within TIE, so that equal centre cells all count.

Measured: how soon the identification is right (exactly the corners and the centre cells, and no others) and stays
right.
"""

from __future__ import annotations

import argparse

import numpy as np
from numba import njit

SIZE = 48
UPDATE = 0.5        # the chance that a cell updates in a tick
JITTER = 0.01       # random jitter in every value a cell computes
RATE = 1.0          # steps: how fast a count moves to its target, per unit time
FAR = 1e6           # steps: a count not yet known
MAKE = 0.5          # leak: chemical made per cell per unit time
SPREAD = 1.0        # leak: how fast a cell moves to its neighbour slots' mean, per unit time
TIE = 0.3           # how near a neighbour's value can be and still count as equal (well above the noise, which
                    # accumulates along counts, and below the real differences at the corners and centre)
NOISE = 0.1         # cells start with values uniform in 0..NOISE

# The cells' numbers: steps uses IN (count in from the edge) and OUT (count out from the centre); leak uses CHEM.
# CENTRE and CORNER are what each cell has decided about itself.
IN, OUT, CHEM, CENTRE, CORNER = range(5)
LAYERS = 5


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _ticks(s, s2, rule, ticks, dt, k):
    """ticks ticks. Each cell, with chance UPDATE, updates from its own numbers and its neighbours' (see the module's
    description). rule 0 is steps, 1 is leak."""
    update, jitter, rate, far, make, spread, tie = k[0], k[1], k[2], k[3], k[4], k[5], k[6]
    n = s.shape[1]
    root2 = np.sqrt(2.0)
    step = min(1.0, rate * dt)
    for _ in range(ticks):
        s2[:, :, :] = s
        for y in range(n):
            for x in range(n):
                if np.random.random() >= update:
                    continue
                edge = False                                    # a neighbour slot beyond the edge
                least_in, least_out = far, far                  # smallest neighbour count plus the step
                most_in, most_out = -far, -far                  # largest neighbour count
                chem_sum, chem_most, chem_least = 0.0, -far, far
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if not (dy or dx):
                            continue
                        yy, xx = y + dy, x + dx
                        if not (0 <= yy < n and 0 <= xx < n):
                            edge = True
                            continue
                        cost = root2 if dy and dx else 1.0
                        # counts are read as whole numbers, so noise under half a step is wiped out at every step
                        # rather than piling up along the count
                        least_in = min(least_in, np.round(s[IN, yy, xx]) + 1.0)
                        least_out = min(least_out, np.round(s[OUT, yy, xx]) + (2.0 if dy and dx else 1.0))   # rook
                        most_in = max(most_in, np.round(s[IN, yy, xx]))
                        most_out = max(most_out, np.round(s[OUT, yy, xx]))
                        chem_sum += s[CHEM, yy, xx]
                        chem_most = max(chem_most, s[CHEM, yy, xx])
                        chem_least = min(chem_least, s[CHEM, yy, xx])
                if rule == 0:
                    # counts jump to their target (with a chance matching the rate, so the same speed at any tick
                    # size) rather than creeping, so they stay whole numbers
                    target = 0.0 if edge else least_in          # in from the edge
                    if np.random.random() < step:
                        s2[IN, y, x] = target + jitter * np.random.standard_normal()
                    centre = (not edge) and np.round(s[IN, y, x]) >= most_in
                    target = 0.0 if centre else least_out       # out from the centre
                    if np.random.random() < step:
                        s2[OUT, y, x] = target + jitter * np.random.standard_normal()
                    corner = edge and np.round(s[OUT, y, x]) >= most_out
                else:
                    mean = chem_sum / 8.0                       # the eight slots; beyond the edge holds none
                    c = s[CHEM, y, x]
                    s2[CHEM, y, x] = c + min(1.0, spread * dt) * (mean - c) + make * dt + jitter * np.random.standard_normal()
                    centre = c >= chem_most - tie
                    corner = c < chem_least - tie               # a sharp dip: the edge middles are flat
                s2[CENTRE, y, x] = 1.0 if centre else 0.0
                s2[CORNER, y, x] = 1.0 if corner else 0.0
        s[:, :, :] = s2


def knobs() -> np.ndarray:
    return np.array([UPDATE, JITTER, RATE, FAR, MAKE, SPREAD, TIE])


def life(seed: int = 0, dt: float = 1.0, rule: str = "steps"):
    """One life, without end: yields (time, the cells' numbers) after every tick (the same array, changed in place)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, NOISE, (LAYERS, SIZE, SIZE))     # all cells the same, bar noise
    s[CENTRE], s[CORNER] = 0.0, 0.0
    s2 = np.empty_like(s)
    k = knobs()
    _seed(seed)
    step = 0
    while True:
        _ticks(s, s2, 0 if rule == "steps" else 1, 1, dt, k)
        step += 1
        yield step * dt, s


def truth(size: int = None):
    """The right answer: the four corners, and the centre cells (one, or four for an even size)."""
    n = size or SIZE
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    mid = [(n - 1) // 2, n // 2]
    centre = {(a, b) for a in set(mid) for b in set(mid)}
    return corners, centre


def judge(s: np.ndarray):
    """(right, wrong): right if exactly the corners say they are corners and exactly the centre cells say they are
    the centre; and how many cells are wrong either way."""
    corners, centre = truth()
    said_corner = {tuple(c) for c in np.argwhere(s[CORNER] > 0.5)}
    said_centre = {tuple(c) for c in np.argwhere(s[CENTRE] > 0.5)}
    wrong = len(said_corner ^ corners) + len(said_centre ^ centre)
    return wrong == 0, wrong


def run(seed=0, dt=1.0, rule="steps", time=3000.0, every=10.0):
    """When the identification first becomes right, and how much of the time after that it stays right."""
    first, right_after, checks = None, 0, 0
    for t, s in life(seed, dt, rule):
        if abs(t / every - round(t / every)) < 1e-9:
            ok, wrong = judge(s)
            if first is None and ok:
                first = t
            if first is not None:
                checks += 1
                right_after += ok
        if t >= time - dt / 2:
            return first, (right_after / checks if checks else 0.0), judge(s)[1]


def picture(s: np.ndarray, rule: str) -> np.ndarray:
    """In greys: the cells' value (dim) and the cells that say they are a corner or the centre (white)."""
    v = s[IN] if rule == "steps" else s[CHEM]
    v = (v - v.min()) / (np.ptp(v) or 1.0)
    shade = 0.15 + 0.45 * v
    return np.where((s[CENTRE] > 0.5) | (s[CORNER] > 0.5), 1.0, shade)


def save_picture(pictures, path: str, scale: int = 4) -> None:
    import pygame
    strip = np.concatenate([np.pad(p, ((0, 0), (0, 2)), constant_values=0.5) for p in pictures], axis=1)
    grey = (255 * strip).astype(np.uint8).repeat(scale, 0).repeat(scale, 1)
    pygame.image.save(pygame.surfarray.make_surface(np.stack([grey.T] * 3, axis=-1)), path)


def watch(seed: int = 0, dt: float = 1.0, rule: str = "steps", scale: int = 14, title: str = "places") -> None:
    """A life, live, in greys: the cells' value dim, the cells that say they are a corner or the centre white, and
    lettered (K corner, C centre). Keys: r restarts; space pauses; up and down arrows change how many ticks pass per
    frame; Escape quits."""
    import pygame
    pygame.init()
    side = SIZE * scale
    screen = pygame.display.set_mode((side, side + 50))
    pygame.display.set_caption(title)
    font = pygame.font.SysFont(None, 20)
    ticks_per_frame, paused = 1, False
    clock = pygame.time.Clock()
    lives = life(seed, dt, rule)
    while True:
        restart = False
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
                    lives, restart = life(seed, dt, rule), True
        if paused and not restart:
            clock.tick(30)
            continue
        for _ in range(ticks_per_frame):
            t, s = next(lives)
        grey = (255 * picture(s, rule)).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        for kind, letter in ((CORNER, "K"), (CENTRE, "C")):
            for r, c in np.argwhere(s[kind] > 0.5):
                screen.blit(font.render(letter, True, (0, 0, 0)), (c * scale + 2, r * scale))
        ok, wrong = judge(s)
        status = f"{rule}   time {t:7.0f}   {'right' if ok else f'{wrong} cells wrong'}   {ticks_per_frame} ticks per frame"
        screen.blit(font.render(status, True, (255, 255, 255)), (6, side + 6))
        screen.blit(font.render("r restart   space pause   up/down speed   Esc quit", True, (255, 255, 255)), (6, side + 28))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=("steps", "leak"), default="steps")
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=3000.0)
    parser.add_argument("--clean", action="store_true", help="no jitter")
    parser.add_argument("--picture", help="save the development as a PNG")
    parser.add_argument("--watch", action="store_true", help="watch it live instead")
    args = parser.parse_args()
    SIZE = args.size
    if args.clean:
        JITTER = 0.0
    if args.watch:
        watch(args.seed, args.dt, args.rule, title=f"places: {args.rule}")
        raise SystemExit
    first, stays, wrong = run(args.seed, args.dt, args.rule, args.time)
    print(f"{args.rule}: right first at {first}, right {100 * stays:.0f}% of the time after, {wrong} cells wrong at the end")
    if args.picture:
        pics, marks = [], [args.time * f for f in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)]
        for t, s in life(args.seed, args.dt, args.rule):
            if marks and t >= marks[0] - args.dt / 2:
                pics.append(picture(s, args.rule)); marks.pop(0)
            if not marks:
                break
        save_picture(pics, args.picture)
