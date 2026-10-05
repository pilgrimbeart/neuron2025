"""Growing a path: can a bare sheet wire a source to a sink by itself? (README, "Next goal"; LESSONS.md, "Growing
wiring".)

    python paths.py [--seed N] [--dt X] [--time T] [--sinks 1|2] [--drains 2,1] [--move source|sink TIME ROW]
                    [--picture FILE.png] [--watch]

A square sheet of identical cells, each holding two numbers:
  c   a quiet chemical. The source cell is kicked (pulses at random times, RATE per unit time, KICK each); each sink
      cell drains it (losing DRAIN x c per unit time, a sink with drain 2 twice as hard). Between cells it spreads
      through conductance: a cell gains SPREAD x g x (the mean over its neighbours of g' x (c' - c)), so it passes
      only between cells that both conduct.
  g   conductance, 0..1, starting near 1 everywhere (a dense sheet, to be pruned). It grows with the flow through the
      cell and decays without it: g += (GROW x f(q) - DECAY x g) x dt, where q is the flow, and f(q) = q^POWER /
      (HALF^POWER + q^POWER) rises faster than in proportion at first (POWER above 1), so paths carrying more take
      flow from the rest (slime mould's rule, Tero et al. 2007).
The flow through a cell is q = g x sqrt(mean over neighbours of g' x (c' - c)^2): large where c is steep around it in
any direction, so a cell needs no sense of direction or position. A cell sees only its own numbers and the (weighted)
means of a few of them over its up to 8 neighbours (g', g'c', g'c'^2), and the edge of the sheet is just missing neighbours.

--move makes the source (or the first sink) jump to another row of its edge at a given time, to see whether the
wiring follows. Measured at the end: whether source and sink are joined through conducting cells (g above HALF_ON), how much of the
sheet conducts, and how long the path is against a straight line (shortest route through conducting cells).
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

SIZE = 48
SPREAD = 1.0        # how fast c spreads through conducting cells
RATE = 0.5          # source pulses per unit time
KICK = 1.0          # c added per pulse
DRAIN = 0.5         # the sink's drain per unit time, x c
GROW = 1.5e-4       # g's growth per unit time at full flow
DECAY = 1e-4        # g's decay per unit time: slow, so that c spreads across the sheet before g has changed
POWER = 3.0         # how much faster than in proportion flow reinforces
HALF = 0.25         # the flow at which growth is half its full rate
NOISE = 0.05        # g starts at 1 - uniform(0, NOISE)
FLOOR = 0.1         # g never falls below this: a cell that has stopped conducting still leaks a little
DIAGONAL = 0.25     # a diagonal neighbour's weight, against 1 for the four beside a cell (the isotropic choice)
HALF_ON = 0.5       # a cell conducts when g is above this
SOURCE = (SIZE // 2, SIZE - 3)                                      # (row, column): by the right edge
SINKS = {1: ((SIZE // 2, 2),), 2: ((SIZE // 4, 2), (3 * SIZE // 4, 2))}   # by the left edge


@dataclass(frozen=True)
class Ends:
    """Where the source and sinks are, how hard each sink drains, and optionally a move: at a given time the source
    (or the first sink) jumps to another cell."""
    source: tuple = SOURCE
    sinks: tuple = SINKS[1]
    drains: tuple = (1.0, 1.0)
    move: tuple | None = None       # (time, "source" or "sink", (row, column))

    def at(self, time: float) -> "Ends":
        """The ends in place at this time."""
        if self.move is None or time < self.move[0]:
            return self
        _, what, cell = self.move
        if what == "source":
            return replace(self, source=cell, move=None)
        return replace(self, sinks=(cell,) + self.sinks[1:], move=None)


def neighbour_sum(x: np.ndarray) -> np.ndarray:
    """Each cell's weighted sum of x over its up to 8 neighbours (missing ones beyond the edge count as nothing):
    the 4 beside it weigh 1, the 4 diagonal ones DIAGONAL, so that spreading doesn't favour the diagonals."""
    p = np.pad(x, 1)
    return sum((DIAGONAL if dy and dx else 1.0) * p[1 + dy:1 + dy + SIZE, 1 + dx:1 + dx + SIZE]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))


COUNT = neighbour_sum(np.ones((SIZE, SIZE)))


def mean(x: np.ndarray) -> np.ndarray:
    return neighbour_sum(x) / COUNT


def flow(c: np.ndarray, g: np.ndarray) -> np.ndarray:
    """q: g x the root mean of g' (c' - c)^2 over the neighbours, from the means of g', g'c' and g'c'^2."""
    square = mean(g * c * c) - 2 * c * mean(g * c) + c * c * mean(g)
    return g * np.sqrt(np.maximum(square, 0.0))


def life(seed: int = 0, dt: float = 0.5, ends: Ends = Ends()):
    """One life, tick by tick, without end: yields (time, g, c, the ends in place) after each tick (the same arrays,
    changed in place)."""
    rng = np.random.default_rng(seed)
    g = 1.0 - rng.uniform(0, NOISE, (SIZE, SIZE))
    c = np.zeros((SIZE, SIZE))
    step, now = 0, None
    while True:
        if ends.at(step * dt) != now:
            now = ends.at(step * dt)
            drain = np.zeros((SIZE, SIZE))
            for cell, strength in zip(now.sinks, now.drains):
                drain[cell] = DRAIN * strength
        c += SPREAD * g * (mean(g * c) - c * mean(g)) * dt
        c -= drain * c * dt
        c[now.source] += KICK * rng.poisson(RATE * dt)
        q = flow(c, g)
        g += (GROW * q ** POWER / (HALF ** POWER + q ** POWER) - DECAY * g) * dt
        np.clip(g, FLOOR, 1.0, out=g)
        step += 1
        yield step * dt, g, c, now


def grow(seed: int = 0, dt: float = 0.5, time: float = 300000.0, ends: Ends = Ends(), snapshots=8):
    """A life of the given length. Returns (g at the end, c at the end, the ends in place at the end, pictures of g at
    evenly spaced moments with the ends marked mid-grey)."""
    steps = int(round(time / dt))
    pictures = []
    for step, (_, g, c, now) in zip(range(steps), life(seed, dt, ends)):
        if (step + 1) % max(1, steps // snapshots) == 0:
            picture = g.copy()
            for cell in (now.source,) + now.sinks:
                picture[cell] = 0.5
            pictures.append(picture)
    return g, c, now, pictures


def route(g: np.ndarray, start, end):
    """The shortest route from start to end through conducting cells (8-connected), as a number of steps; None if
    they aren't joined."""
    on = g > HALF_ON
    if not (on[start] and on[end]):
        return None
    seen = {start: 0}
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            return seen[end]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n = (y + dy, x + dx)
                if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and on[n] and n not in seen:
                    seen[n] = seen[(y, x)] + 1
                    queue.append(n)
    return None


def report(g: np.ndarray, ends: Ends) -> str:
    parts = [f"conducting {100 * (g > HALF_ON).mean():5.1f}% of cells"]
    for end in ends.sinks:
        steps = route(g, ends.source, end)
        straight = max(abs(end[0] - ends.source[0]), abs(end[1] - ends.source[1]))
        parts.append(f"to sink {end}: " + (f"joined, path {steps} steps ({steps / straight:.2f} x straight)"
                                           if steps is not None else "not joined"))
    return "; ".join(parts)


def save_pictures(pictures, path: str, scale: int = 4) -> None:
    """The pictures side by side, in grey (0 black, 1 white)."""
    import pygame
    strip = np.concatenate([np.pad(p, ((0, 0), (0, 2)), constant_values=0.5) for p in pictures], axis=1)
    grey = (255 * strip).astype(np.uint8).repeat(scale, 0).repeat(scale, 1)
    surface = pygame.surfarray.make_surface(np.stack([grey.T] * 3, axis=-1))
    pygame.image.save(surface, path)


def watch(seed: int = 0, dt: float = 0.5, ends: Ends = Ends(), scale: int = 14) -> None:
    """A life, live: g in grey (0 black, 1 white), the source lettered S and the sinks K. Space pauses; up and down
    arrows change how many ticks pass per frame; Escape quits."""
    import pygame
    pygame.init()
    side = SIZE * scale
    screen = pygame.display.set_mode((side, side + 30))
    pygame.display.set_caption("paths: growing a path")
    font = pygame.font.SysFont(None, 24)
    ticks_per_frame, paused = 200, False
    clock = pygame.time.Clock()
    for time, g, c, now in life(seed, dt, ends):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    ticks_per_frame *= 2
                if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    ticks_per_frame = max(1, ticks_per_frame // 2)
            if not paused:
                break
            clock.tick(30)
        if int(round(time / dt)) % ticks_per_frame:
            continue
        grey = (255 * g).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        for (r, k), letter in [(now.source, "S")] + [(end, "K") for end in now.sinks]:
            label = font.render(letter, True, (255, 255, 255), (90, 90, 90))
            screen.blit(label, (k * scale, r * scale))
        status = (f"time {time:9.0f}   conducting {100 * (g > HALF_ON).mean():5.1f}%   {ticks_per_frame} ticks per frame"
                  f"   space: pause, up/down: speed, Esc: quit")
        screen.blit(font.render(status, True, (255, 255, 255)), (6, side + 6))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--time", type=float, default=300000.0)
    parser.add_argument("--sinks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--drains", default="1,1", help="the sinks' drain strengths, e.g. 2,1")
    parser.add_argument("--move", nargs=3, metavar=("WHAT", "TIME", "ROW"),
                        help="at TIME, the source (or the first sink) jumps to ROW of its edge, e.g. sink 300000 8")
    parser.add_argument("--picture", help="save g's development as a PNG")
    parser.add_argument("--watch", action="store_true", help="watch it live instead")
    args = parser.parse_args()
    ends = Ends(sinks=SINKS[args.sinks], drains=tuple(map(float, args.drains.split(","))))
    if args.move:
        what, time, row = args.move
        column = (ends.source if what == "source" else ends.sinks[0])[1]
        ends = replace(ends, move=(float(time), what, (int(row), column)))
    if args.watch:
        watch(args.seed, args.dt, ends)
        raise SystemExit
    g, c, now, pictures = grow(args.seed, args.dt, args.time, ends)
    print(report(g, now))
    if args.picture:
        save_pictures(pictures, args.picture)
