"""Growing wiring: can a bare sheet wire its sources to its sinks by itself? (README, "Next goal"; LESSONS.md,
"Growing paths". The earlier, slow flow rule is at git tag exp/paths-flow.)

    python paths.py [--sources R,C ...] [--sinks R,C ...] [--seed N] [--dt X] [--time T] [--clean]
                    [--move source|sink TIME ROW COLUMN] [--picture FILE.png] [--watch [--title T]]

Two tricks:
  1. Sinks send out waves, which count their steps.
  2. Each cell remembers which neighbour the wave came from (its breadcrumb). A source is on the path, and so is any
     cell a path cell's breadcrumb points to: a path is the trail of breadcrumbs from a source to the nearest sink.
When the waves stop, cells forget their breadcrumbs, and the path vanishes.

And two more, so that a wired sheet falls quiet and the wiring stays (PERSIST):
  3. A sink calls (sends waves) only while it is hungry: no pulse has reached it for a while (HUNGER). Pulses run
     along paths (a source kick sends one; each path cell passes it to the cell its breadcrumb names), so once a path
     delivers, its sink stops calling, and the sheet goes quiet.
  4. A path lives as long as its sink thanks it: a sink that receives a pulse sends an acknowledgement back up the
     path, which refreshes the breadcrumbs as waves do (the retrograde growth factor that keeps a neuron's
     connection alive). A path to a vanished sink is no longer thanked, and is forgotten. And a cell thanked within
     KEEP keeps its breadcrumb whatever waves pass: if it works, it isn't changed.
Subtext for these: a source not thanked for a while (lonely) sends out a bare request wave, and a sink it reaches
calls for CALL time, so new and stranded sources get wired even when the sinks are fed; pulses and acknowledgements
are passed for PULSE_TIME and then rest only briefly (PULSE_REST: a longer rest let bunched-up pulses die).

The subtext, for a noisy world:
  - Waves: a cell next to an excited one becomes excited for EXCITED time, carrying the smallest of its excited
    neighbours' counts plus the step (1, or sqrt 2 diagonally, so that space is round), then rests for REFRACTORY
    time, so waves only travel outwards, and collide and vanish. EXCITED is long enough that a cell rarely misses a
    wave even though it updates at random times, and REFRACTORY long enough that one that did can't send a wave back
    into cells ready again (re-entry). The count must be carried: with random update times, the order in which
    neighbours are woken is close to a coin toss, but the count isn't affected.
  - Breadcrumbs are settled when the wave has passed: while excited, a cell watches the ways the wave comes; it keeps
    its breadcrumb if the wave came through it (with a smaller count) nearly as short as the best way (within
    MARGIN), and otherwise takes the best way. So breadcrumbs change only when the waves stop coming that way (a sink
    moved, or a shorter way opened), not with noise, and they never loop. Each cell's breadcrumb names a neighbour by
    its tag, a random name it is born with (nothing about position).
  - Forgetting: a cell that has had no wave for EXPIRE time forgets its breadcrumb.
  - Ends are kicked at random times by the world (RATE per unit time, as sensors are); kicks top up a hold that fades
    (HOLD_FADE), and a cell is a source (or sink) while its hold is above 1/2; a sink kick also sends a wave.
  - Every cell updates at random times (each tick with chance UPDATE), and each count gets random jitter (JITTER;
    --clean turns it off).

Measured: which pairs of a source and a sink are joined through path cells, and how much of the sheet is path.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from numba import njit, prange

SIZE = 48
RATE = 0.5          # end kicks per unit time
HOLD_FADE = 0.05    # how fast an end cell's hold fades per unit time (kicks top it up)
EXCITED = 16.0      # how long a cell stays excited when a wave passes
REFRACTORY = 40.0   # how long it is then refractory (longer than EXCITED, so a wave can't turn back)
EXPIRE = 200.0      # how long a cell keeps its breadcrumb without a wave (more than a wave period)
FAR = 1e9           # no count known (in effect infinite: a count may be as long as the sheet is big)
UPDATE = 0.5        # the chance that a cell updates in a tick (never all at once)
JITTER = 0.01       # random jitter in each count a wave carries
MARGIN = 0.1        # how much longer (a fraction) the way through its breadcrumb may be before a cell switches
PERSIST = True      # pulses along paths, acknowledgements back, and sinks that send waves only when hungry
PULSE_TIME = 16.0   # how long a cell passes a pulse (or an acknowledgement) on: long enough for random updates
PULSE_REST = 2.0   # how long it then rests from passing another
CALL = 150.0        # how long a sink keeps calling (sending waves) after a request reaches it, fed or not
KEEP = 50.0         # a cell thanked within this long keeps its breadcrumb whatever waves pass (in use: leave it)
HUNGER = 200.0      # a sink sends waves only after this long without receiving a pulse
HALF_ON = 0.5       # a cell is path when p is above this
KICK_SENDS = True   # an end kick sends a pulse down the path (False: pulses only when the world says, as a sensor)


@dataclass(frozen=True)
class Ends:
    """Where the sources and sinks are, and optionally a move: at a given time the first source (or sink) jumps to
    another cell."""
    sources: tuple = ((SIZE // 2, SIZE - 3),)       # (row, column): by the right edge
    sinks: tuple = ((SIZE // 2, 2),)                # by the left edge
    move: tuple | None = None                       # (time, "source" or "sink", (row, column))
    cuts: tuple = ()                                # cells that can't be on a path (waves still pass)
    walls: tuple = ()                               # cells that waves can't pass (so paths go round)

    def at(self, time: float) -> "Ends":
        """The ends in place at this time."""
        if self.move is None or time < self.move[0]:
            return self
        _, what, cell = self.move
        if what == "source":
            return replace(self, sources=(cell,) + self.sources[1:], move=None)
        return replace(self, sinks=(cell,) + self.sinks[1:], move=None)


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


def knobs() -> np.ndarray:
    """The rule's numbers, as `_ticks` takes them (read when a life starts)."""
    return np.array([RATE, HOLD_FADE, EXCITED, REFRACTORY, EXPIRE, FAR, UPDATE, JITTER, MARGIN,
                     1.0 if PERSIST else 0.0, PULSE_TIME, PULSE_REST, HUNGER, CALL, KEEP, 1.0 if KICK_SENDS else 0.0])


# The cells' numbers, one layer each: the wave (excited and refractory time left, the count it carried here), the
# breadcrumb (the tag of the neighbour the wave came from) and what the cell watched while excited (the count via its
# breadcrumb, the best count and whose), when a wave last came, the ends' holds, and whether it is on the path.
# And, for persistence: a pulse passing along the path (firing and resting time left), an acknowledgement passing back
# up it (the same), the time since a sink last received a pulse, and how many pulses it has received.
# And the request waves lonely sources send (excited and resting time left), and how long since a source was thanked.
(EXCITED_LEFT, RESTING_LEFT, COUNT, CRUMB, VIA_CRUMB, BEST, BEST_CRUMB, AGE, SOURCE, SINK, PATH,
 FIRE_LEFT, FIRE_REST, ACK_LEFT, ACK_REST, FED, ARRIVED, ASK_LEFT, ASK_REST, LONELY, CALLING, THANKED) = range(22)
LAYERS = 22


@njit(cache=True, parallel=True)
def _ticks(s, s2, tag, is_source, is_sink, is_cut, is_wall, send, ticks, dt, k):
    """ticks ticks. s holds the cells' numbers (the layers above), s2 is scratch, tag is each cell's name,
    is_source and is_sink mark the cells the world kicks, and send marks sources the world makes send a pulse now. First every cell's clocks run and the ends get their kicks;
    then each cell, with chance UPDATE, updates from its own numbers and its neighbours' (see the module's
    description)."""
    rate, fade, excited, refractory, expire, far, update, jitter, margin = (
        k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8])
    persist, pulse_time, pulse_rest, hunger, call, keep = k[9] > 0.5, k[10], k[11], k[12], k[13], k[14]
    kick_sends = k[15] > 0.5
    n = s.shape[1]
    root2 = np.sqrt(2.0)
    for _ in range(ticks):
        for y in prange(n):                             # rows in parallel (each cell reads the old state)
            for x in range(n):
                s[AGE, y, x] += dt
                if s[EXCITED_LEFT, y, x] > 0.0:
                    s[EXCITED_LEFT, y, x] -= dt
                    if s[EXCITED_LEFT, y, x] <= 0.0:                # the wave has passed: rest, and settle the
                        s[EXCITED_LEFT, y, x] = 0.0                 # breadcrumb (kept unless the best way was
                        s[RESTING_LEFT, y, x] = refractory          # clearly shorter; a sink has none)
                        if (s[VIA_CRUMB, y, x] > s[BEST, y, x] * (1.0 + margin) and s[BEST_CRUMB, y, x] >= 0.0
                                and s[SINK, y, x] <= 0.5 and (not persist or s[THANKED, y, x] > keep)):
                            s[CRUMB, y, x] = s[BEST_CRUMB, y, x]
                elif s[RESTING_LEFT, y, x] > 0.0:
                    s[RESTING_LEFT, y, x] = max(0.0, s[RESTING_LEFT, y, x] - dt)
                s[FED, y, x] += dt
                s[LONELY, y, x] += dt
                s[CALLING, y, x] = max(0.0, s[CALLING, y, x] - dt)
                s[THANKED, y, x] += dt
                if s[ASK_LEFT, y, x] > 0.0:                         # request waves: excited, then resting
                    s[ASK_LEFT, y, x] -= dt
                    if s[ASK_LEFT, y, x] <= 0.0:
                        s[ASK_LEFT, y, x], s[ASK_REST, y, x] = 0.0, refractory
                elif s[ASK_REST, y, x] > 0.0:
                    s[ASK_REST, y, x] = max(0.0, s[ASK_REST, y, x] - dt)
                for left, rest in ((FIRE_LEFT, FIRE_REST), (ACK_LEFT, ACK_REST)):   # pulses and acknowledgements
                    if s[left, y, x] > 0.0:                                         # pass, then rest
                        s[left, y, x] -= dt
                        if s[left, y, x] <= 0.0:
                            s[left, y, x], s[rest, y, x] = 0.0, pulse_rest
                    elif s[rest, y, x] > 0.0:
                        s[rest, y, x] = max(0.0, s[rest, y, x] - dt)
                s[SOURCE, y, x] *= 1.0 - fade * dt                  # the ends' holds fade; kicks top them up
                s[SINK, y, x] *= 1.0 - fade * dt
                kicked = is_source[y, x] and np.random.random() < rate * dt
                if (persist and (send[y, x] or (kicked and kick_sends)) and s[PATH, y, x] > 0.5
                        and s[FIRE_LEFT, y, x] == 0.0 and s[FIRE_REST, y, x] == 0.0):
                    s[FIRE_LEFT, y, x] = pulse_time                 # a source sends a pulse down the path
                if kicked:
                    s[SOURCE, y, x] += 1.0
                    if (persist and s[LONELY, y, x] > hunger and s[ASK_LEFT, y, x] == 0.0
                            and s[ASK_REST, y, x] == 0.0):              # a lonely source asks, with a request wave
                        s[ASK_LEFT, y, x] = excited
                if is_sink[y, x] and np.random.random() < rate * dt:
                    s[SINK, y, x] += 1.0
                    if (s[EXCITED_LEFT, y, x] == 0.0 and s[RESTING_LEFT, y, x] == 0.0
                            and (not persist or s[FED, y, x] > hunger or s[CALLING, y, x] > 0.0)):
                        # a sink's kick sends a wave while it is hungry, or calling because it was asked
                        s[EXCITED_LEFT, y, x], s[COUNT, y, x], s[AGE, y, x] = excited, 0.0, 0.0
                        s[CRUMB, y, x], s[VIA_CRUMB, y, x], s[BEST, y, x], s[BEST_CRUMB, y, x] = -1.0, 0.0, far, -1.0
        s2[:, :, :] = s
        for y in prange(n):                             # rows in parallel (each cell reads the old state)
            for x in range(n):
                if np.random.random() >= update:
                    continue
                best, best_crumb = far, -1.0                # the best way the wave comes: count + step, and whose
                via_crumb = far                             # the way it comes through my breadcrumb
                named = False                               # a path cell's breadcrumb points to me
                fired_at = False                            # ... and it is passing me a pulse
                acked = False                               # the cell my breadcrumb names is passing back an ack
                asked = False                               # a neighbour is passing on a request wave
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = y + dy, x + dx
                        if not ((dy or dx) and 0 <= yy < n and 0 <= xx < n):
                            continue
                        if s[EXCITED_LEFT, yy, xx] > 0.0:
                            v = s[COUNT, yy, xx] + (root2 if dy and dx else 1.0)
                            if v < best:
                                best, best_crumb = v, tag[yy, xx]
                            if tag[yy, xx] == s[CRUMB, y, x] and s[COUNT, yy, xx] < s[COUNT, y, x]:
                                via_crumb = v
                        if s[PATH, yy, xx] > 0.5 and s[CRUMB, yy, xx] == tag[y, x]:
                            named = True
                            if s[FIRE_LEFT, yy, xx] > 0.0:
                                fired_at = True
                        if tag[yy, xx] == s[CRUMB, y, x] and s[ACK_LEFT, yy, xx] > 0.0:
                            acked = True
                        if s[ASK_LEFT, yy, xx] > 0.0:
                            asked = True
                if s[EXCITED_LEFT, y, x] > 0.0:             # excited: take a smaller count, and watch the ways
                    if best < s[COUNT, y, x] and s[COUNT, y, x] > 0.0:
                        s2[COUNT, y, x] = best
                    if via_crumb < s[VIA_CRUMB, y, x]:
                        s2[VIA_CRUMB, y, x] = via_crumb
                    if best < s[BEST, y, x]:
                        s2[BEST, y, x], s2[BEST_CRUMB, y, x] = best, best_crumb
                elif s[RESTING_LEFT, y, x] == 0.0 and best < far and not is_wall[y, x]:   # ready, next to the wave
                    s2[EXCITED_LEFT, y, x] = excited
                    s2[COUNT, y, x] = best + jitter * np.random.standard_normal()
                    s2[AGE, y, x] = 0.0
                    s2[VIA_CRUMB, y, x], s2[BEST, y, x], s2[BEST_CRUMB, y, x] = via_crumb, best, best_crumb
                if persist and s[PATH, y, x] > 0.5:
                    if fired_at and s[FIRE_LEFT, y, x] == 0.0 and s[FIRE_REST, y, x] == 0.0:
                        s2[FIRE_LEFT, y, x] = pulse_time            # the pulse passes on
                        if s[SINK, y, x] > 0.5:                     # a sink fed: it acknowledges
                            s2[FED, y, x], s2[ARRIVED, y, x] = 0.0, s[ARRIVED, y, x] + 1.0
                            s2[ACK_LEFT, y, x] = pulse_time
                    if acked and s[ACK_LEFT, y, x] == 0.0 and s[ACK_REST, y, x] == 0.0:
                        s2[ACK_LEFT, y, x] = pulse_time             # the acknowledgement passes back up,
                        s2[AGE, y, x] = 0.0                         # refreshing the breadcrumb as a wave would
                        s2[THANKED, y, x] = 0.0                     # (and while thanked, it keeps it: in use)
                        if s[SOURCE, y, x] > 0.5:
                            s2[LONELY, y, x] = 0.0                  # a source thanked is not lonely
                if persist and asked and s[ASK_LEFT, y, x] == 0.0 and s[ASK_REST, y, x] == 0.0 and not is_wall[y, x]:
                    s2[ASK_LEFT, y, x] = excited                    # the request wave passes on; a sink it
                    if s[SINK, y, x] > 0.5:                         # reaches calls, for a while
                        s2[CALLING, y, x] = call
                if s[AGE, y, x] >= expire and s2[AGE, y, x] >= expire:  # no wave for a long while: forget
                    s2[CRUMB, y, x] = -1.0
                s2[PATH, y, x] = 1.0 if (s[SOURCE, y, x] > 0.5 or named) and not is_cut[y, x] else 0.0
        s[:, :, :] = s2


def born(seed: int = 0):
    """A new sheet: (the cells' numbers, scratch, each cell's tag, the knobs)."""
    rng = np.random.default_rng(seed)
    s = np.zeros((LAYERS, SIZE, SIZE))
    s[COUNT], s[AGE], s[CRUMB], s[VIA_CRUMB], s[BEST], s[BEST_CRUMB] = FAR, EXPIRE + 1.0, -1.0, FAR, FAR, -1.0
    s[FED] = 1e9                                    # sinks are born hungry
    s[LONELY] = 1e9                                 # sources are lonely from birth: they ask until first thanked,
                                                    # however long a big sheet makes the first round trip
    s[THANKED] = 1e9                                # nothing has been thanked
    tag = rng.random((SIZE, SIZE))                  # each cell's name, from birth: random, nothing about position
    _seed(seed)
    return s, np.empty_like(s), tag, knobs()


def masks(cells) -> np.ndarray:
    """A SIZE x SIZE mask, true at the given cells."""
    m = np.zeros((SIZE, SIZE), np.bool_)
    for cell in cells:
        m[cell] = True
    return m


def life(seed: int = 0, dt: float = 1.0, ends: Ends = Ends()):
    """One life, without end: yields (time, path, the ends in place) after every tick (path is the same array,
    changed in place). Sending it new Ends moves the ends."""
    s, s2, tag, k = born(seed)
    quiet = masks(())
    step = 0
    while True:
        now = ends.at(step * dt)
        _ticks(s, s2, tag, masks(now.sources), masks(now.sinks), masks(now.cuts), masks(now.walls), quiet, 1, dt, k)
        step += 1
        sent = yield step * dt, s[PATH], now
        if sent is not None:                        # new ends, sent in (the live view's keys)
            ends = sent


def grow(seed: int = 0, dt: float = 1.0, time: float = 300.0, ends: Ends = Ends(), snapshots=8):
    """A life of the given length. Returns (p at the end, the ends in place at the end, pictures of p at evenly spaced
    moments with the ends marked mid-grey)."""
    ticks = max(1, int(round(time / dt)))
    pictures = []
    for i, (_, p, now) in zip(range(ticks), life(seed, dt, ends)):
        if (i + 1) % max(1, ticks // snapshots) == 0:
            picture = p.copy()
            for cell in now.sources + now.sinks:
                picture[cell] = 0.5
            pictures.append(picture)
    return p, now, pictures


def route(p: np.ndarray, start, end):
    """The shortest route from start to end through path cells (8-connected), as a number of steps; None if they
    aren't joined."""
    on = p > HALF_ON
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


def report(p: np.ndarray, ends: Ends) -> str:
    joined = sum(route(p, a, b) is not None for a in ends.sources for b in ends.sinks)
    pairs = len(ends.sources) * len(ends.sinks)
    return f"path {100 * (p > HALF_ON).mean():5.1f}% of cells ({(p > HALF_ON).sum()}); {joined}/{pairs} source-sink pairs joined"


def save_pictures(pictures, path: str, scale: int = 4) -> None:
    """The pictures side by side, in grey (0 black, 1 white)."""
    import pygame
    strip = np.concatenate([np.pad(p, ((0, 0), (0, 2)), constant_values=0.5) for p in pictures], axis=1)
    grey = (255 * strip).astype(np.uint8).repeat(scale, 0).repeat(scale, 1)
    surface = pygame.surfarray.make_surface(np.stack([grey.T] * 3, axis=-1))
    pygame.image.save(surface, path)


def watch(seed: int = 0, dt: float = 1.0, ends: Ends = Ends(), scale: int = 14, title: str = "paths") -> None:
    """A life, live, in greys: waves dark grey, the path light grey, pulses and acknowledgements on it white, damage
    mid-grey; sources lettered S and sinks K. Keys at the mouse: s or k adds a source or sink, or, over an end (within
    a cell), removes it; x damages a 3x3 patch (cells that carry neither waves nor path), or repairs it if damaged; r
    restarts the life with the ends as they are; space pauses; up and down arrows change how many ticks pass per
    frame; Escape quits."""
    import pygame
    pygame.init()
    side = SIZE * scale
    screen = pygame.display.set_mode((side, side + 50))
    pygame.display.set_caption(title)
    font = pygame.font.SysFont(None, 24)
    ticks_per_frame, paused = 1, False
    clock = pygame.time.Clock()
    lives = life(seed, dt, ends)
    moved, restarted = None, False
    while True:
        time, p, now = lives.send(moved) if moved else next(lives)
        moved, restarted = None, False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if event.type != pygame.KEYDOWN:
                    continue
                mx, my = pygame.mouse.get_pos()
                cell = (my // scale, mx // scale)
                base = moved or now
                if event.key in (pygame.K_s, pygame.K_k) and 0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE:
                    near = lambda c: max(abs(c[0] - cell[0]), abs(c[1] - cell[1])) <= 1
                    if any(near(c) for c in base.sources + base.sinks):          # over an end: remove it
                        moved = replace(base, move=None, sources=tuple(c for c in base.sources if not near(c)),
                                        sinks=tuple(c for c in base.sinks if not near(c)))
                    else:                                                       # elsewhere: add one
                        kind = "sources" if event.key == pygame.K_s else "sinks"
                        moved = replace(base, move=None, **{kind: getattr(base, kind) + (cell,)})
                if event.key == pygame.K_x and 0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE:   # damage, or repair
                    patch = {(cell[0] + dy, cell[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                    damaged = set(base.walls)
                    damaged = damaged - patch if cell in damaged else damaged | patch
                    moved = replace(base, move=None, walls=tuple(sorted(damaged)), cuts=tuple(sorted(damaged)))
                if event.key == pygame.K_r:                                     # restart, with the ends as they are
                    lives = life(seed, dt, replace(base, move=None))
                    moved = None
                    restarted = True
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    ticks_per_frame *= 2
                if event.key == pygame.K_DOWN:
                    ticks_per_frame = max(1, ticks_per_frame // 2)
            if not paused or moved or restarted:
                break
            clock.tick(30)
        if int(round(time / dt)) % ticks_per_frame:
            continue
        state = lives.gi_frame.f_locals["s"]
        shade = np.where(state[EXCITED_LEFT] > 0, 0.25, 0.0)                # waves
        shade = np.where(p > 0.5, 0.7, shade)                               # the path
        shade = np.where((p > 0.5) & ((state[FIRE_LEFT] > 0) | (state[ACK_LEFT] > 0)), 1.0, shade)   # pulses, thanks
        for c in now.walls:
            shade[c] = 0.45                                                 # damage
        grey = (255 * shade).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        for (r, k), letter in [(c, "S") for c in now.sources] + [(c, "K") for c in now.sinks]:
            label = font.render(letter, True, (255, 255, 255), (90, 90, 90))
            screen.blit(label, (k * scale, r * scale))
        status = f"time {time:7.0f}   {report(p, now)}   {ticks_per_frame} ticks per frame"
        keys = "mouse: s/k add or remove an end, x damage or repair   r restart   space   up/down   Esc"
        screen.blit(font.render(status, True, (255, 255, 255)), (6, side + 6))
        screen.blit(font.render(keys, True, (255, 255, 255)), (6, side + 28))
        pygame.display.flip()
        clock.tick(30)


def cells(text: str) -> tuple:
    """"24,45 8,45" -> ((24, 45), (8, 45))."""
    return tuple(tuple(int(v) for v in c.split(",")) for c in text.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=f"{SIZE // 2},{SIZE - 3}", help='cells, e.g. "8,45 24,45 40,45"')
    parser.add_argument("--sinks", default=f"{SIZE // 2},2", help='cells, e.g. "12,2 36,2"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=300.0)
    parser.add_argument("--move", nargs=4, metavar=("WHAT", "TIME", "ROW", "COLUMN"),
                        help="at TIME, the first source (or sink) jumps to (ROW, COLUMN), e.g. sink 150 6 24")
    parser.add_argument("--picture", help="save p's development as a PNG")
    parser.add_argument("--watch", action="store_true", help="watch it live instead")
    parser.add_argument("--title", default="paths", help="the live view's window title")
    parser.add_argument("--clean", action="store_true", help="no jitter (cells still update at random times)")
    args = parser.parse_args()
    if args.clean:
        JITTER = 0.0
    ends = Ends(sources=cells(args.sources), sinks=cells(args.sinks))
    if args.move:
        what, time, row, column = args.move
        ends = replace(ends, move=(float(time), what, (int(row), int(column))))
    if args.watch:
        watch(args.seed, args.dt, ends, title=args.title)
        raise SystemExit
    p, now, pictures = grow(args.seed, args.dt, args.time, ends)
    print(report(p, now))
    if args.picture:
        save_pictures(pictures, args.picture)
