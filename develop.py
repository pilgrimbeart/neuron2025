"""Development in stages, all by local rules: a sheet of identical cells first lays down a coordinate system, then three
points appear (the corners of an equilateral triangle some way in from the edge), then paths grow between them.

    python develop.py [--size N] [--seed N] [--time T] [--watch]

Each cell runs the frame rule (frame.py) and three copies of the path rule (paths.py), one per side of the triangle.
  1. Coordinates: the frame rule (corners elect an origin, the origin chooses the axes, counts in from the edges give
     x and y as fractions of the body).
  2. Points: a cell whose coordinates are nearest one of the triangle's corners, and whose frame is consistent (its
     counts from opposite edges add up as its neighbours' do), names itself point 1, 2 or 3.
  3. Paths: in channel k, point k is a source and the next point a sink (1 to 2, 2 to 3, 3 to 1); a named point pulses
     at random. The breadcrumb rule, with persistence, grows each side of the triangle and then falls quiet.
No timers hand over between stages: each stage has nothing to work on until the one before has produced it (no cell
is a source or sink until it has a name, and none has a name until its frame is complete). Three channels, because
each path heads for a different point (routing to a particular target costs a channel per target; LESSONS.md,
"Crossing").
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import frame
import paths

SIZE = 49
R = 0.3                                                 # the triangle's corners: this far from the centre (fractions)
POINTS = {str(i + 1): (0.5 + R * np.cos(np.pi / 2 + 2 * np.pi * i / 3), 0.5 + R * np.sin(np.pi / 2 + 2 * np.pi * i / 3))
          for i in range(3)}                            # (x, y): point 1 at the top, then round


def life(seed: int = 0):
    """One life, without end: yields (time, the frame's numbers, the three channels' path maps, the named cells for
    each point) after every tick."""
    frame.SIZE = paths.SIZE = SIZE
    frame.RULE = "waves"
    frame.PLACES = POINTS
    body = frame.life(seed)
    channels = [paths.life(seed + 17 * (k + 1), 1.0, paths.Ends(sources=(), sinks=())) for k in range(3)]
    ends = [((), ()) for _ in range(3)]
    while True:
        t, s, tag = next(body)
        named = [tuple(tuple(int(v) for v in c) for c in np.argwhere(s[frame.NAME] == k + 1)) for k in range(3)]
        maps = []
        for k, lives in enumerate(channels):
            want = (named[k], named[(k + 1) % 3])           # channel k: point k to the next point
            if want != ends[k]:
                ends[k] = want
                _, p, _ = lives.send(paths.Ends(sources=want[0], sinks=want[1]))
            else:
                _, p, _ = next(lives)
            maps.append(p)
        yield t, s, tag, maps, named, channels


def run(seed: int = 0, time_: float = 3000.0):
    """When the frame and points are right, when each side of the triangle is first joined, and how busy the waves
    are at the end."""
    marks = {"points": None, "side 1-2": None, "side 2-3": None, "side 3-1": None}
    for t, s, tag, maps, named, channels in life(seed):
        if marks["points"] is None and all(len(n) == 1 for n in named) and frame.judge(s, tag)[0]:
            marks["points"] = t
        if marks["points"] is not None:
            for k in range(3):
                key = f"side {k + 1}-{(k + 1) % 3 + 1}"
                if marks[key] is None and paths.route(maps[k], named[k][0], named[(k + 1) % 3][0]) is not None:
                    marks[key] = t
        if t >= time_:
            waves = np.mean([(ch.gi_frame.f_locals["s"][paths.EXCITED_LEFT] > 0).mean() for ch in channels])
            joined = [paths.route(maps[k], named[k][0], named[(k + 1) % 3][0]) is not None
                      if named[k] and named[(k + 1) % 3] else False for k in range(3)]
            return marks, joined, waves


def watch(seed: int = 0, scale: int = 0) -> None:
    """A life, live: the coordinate grid (dim checkerboard of eighths), the three paths (white), the points lettered
    1, 2, 3. Keys: r restarts with a new seed; space pauses; up and down arrows change the speed; Escape quits."""
    import pygame
    pygame.init()
    n = SIZE
    scale = scale or max(1, 640 // n)
    side = n * scale
    screen = pygame.display.set_mode((side, side + 50))
    pygame.display.set_caption("develop: frame, then points, then paths")
    font = pygame.font.SysFont(None, 22)
    ticks_per_frame, paused = 2, False
    clock = pygame.time.Clock()
    lives = life(seed)
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
                    lives = life(seed)
        if paused:
            clock.tick(30)
            continue
        started, done = time.time(), 0
        while done < ticks_per_frame and (done == 0 or time.time() - started < 0.2):
            t, s, tag, maps, named, channels = next(lives)
            done += 1
        bands = (np.floor(8 * np.clip(s[frame.X], 0, 0.999)) + np.floor(8 * np.clip(s[frame.Y], 0, 0.999))) % 2
        shade = 0.1 + 0.15 * bands
        shade = np.where(np.max(maps, axis=0) > 0.5, 1.0, shade)
        grey = (255 * shade).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        for k in range(3):
            for r, c in named[k][:3]:
                screen.blit(font.render(str(k + 1), True, (255, 255, 255), (70, 70, 70)), (c * scale, r * scale))
        stage = "frame" if not any(named) else ("points" if not any((m > 0.5).sum() > 3 for m in maps) else "paths")
        joined = sum(paths.route(maps[k], named[k][0], named[(k + 1) % 3][0]) is not None
                     for k in range(3) if named[k] and named[(k + 1) % 3])
        status = f"seed {seed}   time {t:6.0f}   stage: {stage}   sides joined {joined}/3"
        screen.blit(font.render(status, True, (255, 255, 255)), (6, side + 6))
        screen.blit(font.render("r new seed   space pause   up/down speed   Esc quit", True, (255, 255, 255)),
                    (6, side + 28))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time", type=float, default=3000.0)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    SIZE = args.size
    if args.watch:
        watch(args.seed)
        raise SystemExit
    marks, joined, waves = run(args.seed, args.time)
    print(f"right at: {marks}; sides joined at the end {joined}; wave activity at the end {waves:.3f}")
