"""Learning in place, by a three-factor rule: on a bare sheet, a red and a blue sensor are wired by grown paths to an
APPROACH and an AVOID output, and the sheet learns, from tastes in random order, which colour is poison.

    python learn.py [--world choice|line] [--size N] [--seed N] [--dt X] [--blocks N] [--poison red|blue] [--watch]

The sheet (Brain): every cell holds the same layers, its own numbers: a learning layer and two breadcrumb channels
(paths.py), one per output. In each channel both sensors' cells are sources and the output the sink, so four routes
grow (red and blue, to approach and to avoid); a channel per target, so they cross freely. The learning layer holds:
  - a balance b (born near INNATE): each pulse the cell sends goes towards APPROACH with chance b, else AVOID;
  - a trace: 1 when a sensor last kicked the cell, fading with time (TRACE);
  - news: the age of the freshest kick it has heard of (its own, or a neighbour's news plus HOP); a cell that hears
    of a kick fresher than its own forgets its own, and its trace is gone. So of the two inputs only the more
    recently kicked holds a trace (lateral inhibition, as news you can watch spread from one input to the other).
and one rule: a taste delivered to the cell moves b by RATE x taste x its trace, bad tastes negative, scaled so b
stays between 0 and 1. Every cell has it; only cells a sensor kicks and a taste reaches ever use it. The three
factors: pre (the trace: this input was the last active), post (a meal: it happens only after approach, so the taste
arriving is itself the sign that approach won), and the taste.

Two worlds (the body and world, not the sheet):
  - choice: one block at a time, red or blue at random; while it is in view its colour's sensor kicks its input cell
    at random times (SENSE); the first output to receive DECIDE pulses wins. Approach: eaten, and DELAY later tasted
    at both input cells, bad if poison. Avoid: it goes, untasted. The next block comes a random while after the
    pulses in flight have arrived.
  - line: the robot on a line, blocks ahead of it at random distances, often both colours in view at once. Each
    pulse arriving at APPROACH moves it a STEP towards them, each at AVOID a STEP back: behaviour is continuous, no
    vote. A sensor kicks at SENSE x nearness of its colour's nearest block (1 / (1 + (distance / REACH)^2), as the
    robot's sensors). A block reached is eaten and tasted; one left beyond LOST is gone; new ones appear at random.
Nothing waits a fixed time that depends on the size.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit, prange

import paths

SIZE = 49
SENSE = 0.1         # sensor kicks per unit time (at full strength)
DECIDE = 3          # choice: pulses an output must receive first to win
DELAY = 10.0        # from a meal to its taste (the body's)
TRACE = 10.0        # how fast an input's trace fades (per unit time; must outlast DELAY, not the sheet)
RATE = 1.0          # how far one taste moves the balance
INNATE = 0.75       # the balance a cell is born with (above 0.5: drawn to anything it sees)
NOISE = 0.02        # ... give or take this much (every cell its own)
UPDATE = 0.5        # the chance that a cell updates in a tick
HOP = 0.1           # news of a kick ages this much more per step it travels (so a cell's own echo is never fresher)
SHOW = 12.0         # watching only: cells holding news younger than this are outlined
DISGUST = 0.0       # a bad taste makes the cell send pulses towards AVOID for this long (0: no reflex)
RECOIL = 0.0        # ... this often (per unit time)
GAP = 50.0          # choice: mean time between one block's pulses arriving and the next block
STEP = 1.0          # line: how far one pulse at an output moves the robot
REACH = 8.0         # line: how far a block's colour carries
LOST = 24.0         # line: a block further than this is gone
SPAWN = 0.01        # line: new blocks per unit time (at most MOST in the world)
MOST = 2
FLAVOUR = True      # line: a block being eaten kicks its colour's sensor (the food in the mouth, sensed strongly)
COLOURS = ("red", "blue")
OUTPUTS = ("approach", "avoid")

# The learning layer: every cell's own numbers. Its balance, how long since a sensor kicked it (its trace is
# exp(-that / TRACE); forgotten, it is infinite), a taste delivered to it and not yet taken, how long its disgust has
# left, and the news: the age of the freshest kick it has heard of (the one number here its neighbours can read).
BALANCE, KICKED, TASTE, DISGUST_LEFT, NEWS = range(5)
LAYERS = 5
NEVER = 1e9


def knobs() -> np.ndarray:
    return np.array([TRACE, RATE, UPDATE, HOP, DISGUST, RECOIL])


@njit(cache=True, parallel=True)
def _ticks(l, l2, kick, taste, send, dt, k):
    """One tick of the learning layer. kick marks the cells a sensor kicked now, taste is what the body delivers to
    each cell now (0 almost everywhere), send[0] and send[1] get the cells that send a pulse into the approach and
    the avoid channel. Every cell, every tick: ages grow; a kick makes the cell's own kick and its news fresh (age 0)
    and sends a pulse, towards approach with chance its balance; a disgusted cell sends towards avoid now and then.
    Then each cell, with chance UPDATE: it takes its neighbours' freshest news (plus HOP), and if that is fresher than
    its own kick, it forgets its own kick (of two inputs only the more recently kicked keeps a trace: lateral
    inhibition); and it takes any taste waiting, moving its balance by RATE x taste x its trace, kept between 0 and 1;
    a bad one disgusts it."""
    trace, rate, update, hop, disgust, recoil = k[0], k[1], k[2], k[3], k[4], k[5]
    n = l.shape[1]
    for y in prange(n):
        for x in range(n):
            send[0, y, x] = send[1, y, x] = False
            l[TASTE, y, x] += taste[y, x]
            l[KICKED, y, x] = min(NEVER, l[KICKED, y, x] + dt)
            l[NEWS, y, x] = min(NEVER, l[NEWS, y, x] + dt)
            if kick[y, x]:
                l[KICKED, y, x] = l[NEWS, y, x] = 0.0
                send[0 if np.random.random() < l[BALANCE, y, x] else 1, y, x] = True
            if l[DISGUST_LEFT, y, x] > 0.0:
                l[DISGUST_LEFT, y, x] = max(0.0, l[DISGUST_LEFT, y, x] - dt)
                if np.random.random() < recoil * dt:
                    send[1, y, x] = True
    l2[:, :, :] = l
    for y in prange(n):
        for x in range(n):
            if np.random.random() >= update:
                continue
            news = l[NEWS, y, x]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    yy, xx = y + dy, x + dx
                    if (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                        news = min(news, l[NEWS, yy, xx] + hop)
            l2[NEWS, y, x] = news
            if news < l[KICKED, y, x]:                      # someone was kicked after me: forget my kick
                l2[KICKED, y, x] = NEVER
            t = l[TASTE, y, x]
            if t != 0.0:
                b = l[BALANCE, y, x]
                l2[BALANCE, y, x] = b + rate * t * np.exp(-l2[KICKED, y, x] / trace) * (b if t < 0.0 else 1.0 - b)
                l2[TASTE, y, x] = 0.0
                if t < 0.0:
                    l2[DISGUST_LEFT, y, x] = disgust
    l[:, :, :] = l2


def places(n: int) -> dict:
    """The bench: where the sensors' input cells and the outputs are, the two inputs 6 cells apart by the west edge,
    outputs by the east, as fractions of the sheet. (In the creature the sheet names these itself.)"""
    at = lambda fy, fx: (int(round(fy * (n - 1))), int(round(fx * (n - 1))))
    r, c = at(0.5, 0.1)
    return {"red": (r - 3, c), "blue": (r + 3, c), "approach": at(0.35, 0.9), "avoid": at(0.65, 0.9)}


class Brain:
    """The sheet: per cell, its learning layer and two path channels (approach, avoid), all its own numbers. Each tick
    the world says, per cell, which are ends (sources, and each channel's sinks), which a sensor kicks and what taste
    arrives; nothing else goes in, and the world reads only the pulses arriving at cells."""

    def __init__(self, seed: int, dt: float, tag: np.ndarray | None = None):
        paths.SIZE = SIZE
        paths.KICK_SENDS = False                    # pulses only when the learning layer sends them
        self.dt = dt
        rng = np.random.default_rng(seed + 7)
        self.tag = rng.random((SIZE, SIZE)) if tag is None else tag    # each cell's name, from birth
        self.channels = []
        for i in range(2):
            s, s2, _, k = paths.born(seed + 101 * (i + 1))
            self.channels.append((s, s2, k))
        self.l = np.zeros((LAYERS, SIZE, SIZE))
        self.l[BALANCE] = np.clip(INNATE + NOISE * rng.standard_normal((SIZE, SIZE)), 0.0, 1.0)
        self.l[KICKED] = self.l[NEWS] = NEVER
        self.l2 = np.empty_like(self.l)
        self.k = knobs()
        self.send = np.zeros((2, SIZE, SIZE), np.bool_)
        self.none = np.zeros((SIZE, SIZE), np.bool_)
        self.last = np.zeros((2, SIZE, SIZE))
        self.t = 0.0

    def tick(self, sources, sinks, kick=None, taste=None) -> np.ndarray:
        """One tick. sources: the cells that are sources in both channels; sinks: [approach's, avoid's]; kick: the
        cells a sensor kicks; taste: what taste arrives at each cell. Returns, per channel and cell, the pulses that
        arrived in this tick."""
        _ticks(self.l, self.l2, self.none if kick is None else kick,
               np.zeros((SIZE, SIZE)) if taste is None else taste, self.send, self.dt, self.k)
        for i, (s, s2, k) in enumerate(self.channels):
            paths._ticks(s, s2, self.tag, sources, sinks[i], self.none, self.none, self.send[i], 1, self.dt, k)
        self.t += self.dt
        arrived = np.stack([s[paths.ARRIVED] for s, _, _ in self.channels])
        new = arrived - self.last
        self.last = arrived
        return new

    def flying(self) -> bool:
        return any((s[paths.FIRE_LEFT] > 0).any() for s, _, _ in self.channels)


class Bench:
    """The bench for learn.py's worlds: the brain with its ends placed by hand (places), kicked and tasted there."""

    def __init__(self, seed: int, dt: float):
        self.brain = Brain(seed, dt)
        self.where = places(SIZE)
        mask = lambda *names: paths.masks(tuple(self.where[c] for c in names))
        self.sources = mask(*COLOURS)
        self.sinks = [mask(o) for o in OUTPUTS]
        self.cell = {c: mask(c) for c in COLOURS}
        self.pending = np.zeros((SIZE, SIZE))

    @property
    def t(self):
        return self.brain.t

    @property
    def balance(self) -> dict:
        return {c: float(self.brain.l[BALANCE][self.where[c]]) for c in COLOURS}

    def tick(self, kicks=()) -> list:
        kick = paths.masks(tuple(self.where[c] for c in kicks))
        taste, self.pending = self.pending, np.zeros((SIZE, SIZE))
        new = self.brain.tick(self.sources, self.sinks, kick, taste)
        return [float(new[i][self.where[o]]) for i, o in enumerate(OUTPUTS)]

    def taste(self, taste: float) -> None:
        """A taste reaches both input cells."""
        self.pending[self.sources] += taste

    def flying(self) -> bool:
        return self.brain.flying()


def choice(seed: int = 0, dt: float = 1.0, poison: str = "red"):
    """The choice world, without end: yields (brain, the block in view or None, events) after every tick; an event
    is (colour, the output that won, the balance before, time)."""
    brain, rng = Bench(seed, dt), np.random.default_rng(seed)
    block, counts, tastes, events = None, [0, 0], [], []
    wait = rng.exponential(GAP)
    while True:
        kicks = (block,) if block is not None and rng.random() < SENSE * dt else ()
        got = brain.tick(kicks)
        if block is not None:
            counts = [c + g for c, g in zip(counts, got)]
            if max(counts) >= DECIDE:                           # the body acts on the first output to DECIDE
                won = OUTPUTS[int(np.argmax(counts))]
                events.append((block, won, brain.balance[block], brain.t))
                if won == "approach":
                    tastes.append((brain.t + DELAY, -1.0 if block == poison else 1.0))
                block, wait = None, rng.exponential(GAP)
        elif not brain.flying():
            wait -= dt
            if wait <= 0:
                block, counts = COLOURS[rng.integers(2)], [0, 0]
        for due, taste in [x for x in tastes if x[0] <= brain.t]:
            tastes.remove((due, taste))
            brain.taste(taste)
        yield brain, block, events


def line(seed: int = 0, dt: float = 1.0, poison: str = "red"):
    """The line world, without end: yields (brain, the blocks [distance, colour], events) after every tick; an event
    is (colour, "eaten" or "lost", the balances before, time)."""
    brain, rng = Bench(seed, dt), np.random.default_rng(seed)
    blocks, tastes, events, flavour = [], [], [], []
    new = lambda: [rng.uniform(REACH / 2, 2 * REACH), COLOURS[rng.integers(2)]]
    while True:
        if len(blocks) < MOST and rng.random() < SPAWN * dt:
            blocks.append(new())
        kicks, flavour = flavour, []
        for c in COLOURS:
            near = min((d for d, colour in blocks if colour == c), default=None)
            if near is not None and rng.random() < SENSE * dt / (1 + (near / REACH) ** 2) and c not in kicks:
                kicks.append(c)
        got = brain.tick(kicks)
        move = STEP * (got[1] - got[0])                         # towards the blocks on approach, back on avoid
        for b in blocks:
            b[0] += move
        for b in list(blocks):
            if b[0] <= 0.0 or b[0] > LOST:
                blocks.remove(b)
                eaten = b[0] <= 0.0
                events.append((b[1], "eaten" if eaten else "lost", dict(brain.balance), brain.t))
                if eaten:
                    tastes.append((brain.t + DELAY, -1.0 if b[1] == poison else 1.0))
                    if FLAVOUR and b[1] not in flavour:
                        flavour.append(b[1])
        for due, taste in [x for x in tastes if x[0] <= brain.t]:
            tastes.remove((due, taste))
            brain.taste(taste)
        yield brain, blocks, events


WORLDS = {"choice": choice, "line": line}


def run(world: str = "choice", seed: int = 0, dt: float = 1.0, poison: str = "red", blocks: int = 30):
    """A life until `blocks` blocks have been met (decided, or eaten or lost). Returns (events, final balances,
    time)."""
    for brain, _, events in WORLDS[world](seed, dt, poison):
        if len(events) >= blocks:
            return events, dict(brain.balance), brain.t


def summary(events, poison: str) -> str:
    """How often each colour was eaten (approached) in the first and second half of the blocks."""
    half = len(events) // 2
    out = []
    for name, part in (("first half", events[:half]), ("second half", events[half:])):
        rates = []
        for c in COLOURS:
            seen = [e for e in part if e[0] == c]
            got = sum(e[1] in ("approach", "eaten") for e in seen)
            rates.append(f"{c}{' (poison)' if c == poison else ''} {got}/{len(seen)}")
        out.append(f"{name}: eaten " + ", ".join(rates))
    bites = [i for i, e in enumerate(events) if e[0] == poison and e[1] in ("approach", "eaten")]
    out.append(f"poison eaten at blocks {bites}")
    return "; ".join(out)


def watch(world: str = "choice", seed: int = 0, dt: float = 1.0, poison: str = "red") -> None:
    """A life, live, in greys: the approach channel's path light grey, the avoid channel's mid grey, pulses white;
    the inputs lettered R and B (with their balances), the outputs A and V; in the line world, a strip below shows
    the robot (left end) and the blocks ahead (R or B). Keys: r new seed (and poison colour); space pauses; up and
    down arrows change the speed; Escape quits."""
    import pygame
    pygame.init()
    n = SIZE
    scale = max(1, 640 // n)
    side = n * scale
    strip = 40 if world == "line" else 0
    screen = pygame.display.set_mode((side, side + strip + 72))
    pygame.display.set_caption(f"learn ({world}): which colour is poison")
    font = pygame.font.SysFont(None, 22)
    ticks_per_frame, paused = 2, False
    clock = pygame.time.Clock()
    lives = WORLDS[world](seed, dt, poison)
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
                    poison = COLOURS[seed % 2]
                    lives = WORLDS[world](seed, dt, poison)
        if paused:
            clock.tick(30)
            continue
        started, done = time.time(), 0
        while done < ticks_per_frame and (done == 0 or time.time() - started < 0.2):
            brain, view, events = next(lives)
            done += 1
        shade = np.zeros((n, n))
        for i, level in ((1, 0.4), (0, 0.7)):
            s = brain.brain.channels[i][0]
            shade = np.where(s[paths.PATH] > 0.5, np.maximum(shade, level), shade)
        for s, *_ in brain.brain.channels:
            shade = np.where((s[paths.PATH] > 0.5) & (s[paths.FIRE_LEFT] > 0), 1.0, shade)
        grey = (255 * shade).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        seen = {view} if world == "choice" else {c for d, c in view if d < REACH}
        for name, letter in (("red", "R"), ("blue", "B"), ("approach", "A"), ("avoid", "V")):
            r, c = brain.where[name]
            text = letter + (f" {brain.balance[name]:.2f}" if name in brain.balance else "")
            lit = name in seen
            screen.blit(font.render(text, True, (0, 0, 0) if lit else (255, 255, 255),
                                    (255, 255, 255) if lit else (70, 70, 70)), (c * scale - 4, r * scale - 4))
        if world == "line":
            y = side + 10
            pygame.draw.line(screen, (120, 120, 120), (10, y + 10), (side - 10, y + 10))
            pygame.draw.circle(screen, (255, 255, 255), (14, y + 10), 7)
            for d, c in view:
                x = 14 + int(d / LOST * (side - 30))
                screen.blit(font.render(c[0].upper(), True, (0, 0, 0), (255, 255, 255) if c == poison else (150, 150, 150)), (x, y + 2))
        last = events[-6:]
        verb = lambda e: "eat" if e[1] in ("approach", "eaten") else ("avoid" if e[1] == "avoid" else "lost")
        history = "  ".join(f"{e[0][0].upper()}:{verb(e)}" for e in last)
        status = f"seed {seed}   time {brain.t:6.0f}   poison {poison}   blocks {len(events)}"
        base = side + strip
        screen.blit(font.render(status, True, (255, 255, 255)), (6, base + 6))
        screen.blit(font.render("recent: " + history, True, (255, 255, 255)), (6, base + 28))
        screen.blit(font.render("R/B balance = chance a pulse goes to A (approach), else V (avoid)   "
                                "r new seed   space   up/down   Esc", True, (255, 255, 255)), (6, base + 50))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=tuple(WORLDS), default="choice")
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--poison", choices=COLOURS, default="red")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    SIZE = args.size
    if args.watch:
        watch(args.world, args.seed, args.dt, args.poison)
        raise SystemExit
    events, balance, t = run(args.world, args.seed, args.dt, args.poison, args.blocks)
    print(summary(events, args.poison))
    print(f"balances at the end: " + ", ".join(f"{c} {b:.2f}" for c, b in balance.items()) + f"; time {t:.0f}")
