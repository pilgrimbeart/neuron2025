"""Learning in place, by a three-factor rule: on a bare sheet, a red and a blue sensor are wired by grown paths to an
APPROACH and an AVOID output, and the sheet learns, from tastes in random order, which colour is poison.

    python learn.py [--world choice|line] [--size N] [--seed N] [--dt X] [--blocks N] [--poison red|blue] [--watch]

The sheet (Brain): two breadcrumb channels (paths.py), one per output. In each, both sensors are sources and the
output the sink, so four routes grow (red and blue, to approach and to avoid); a channel per target, so they cross
freely. The input cell (where the paths start, and where taste arrives, as in the robot) holds two numbers:
  - its balance b: each pulse it sends goes towards APPROACH with chance b, else towards AVOID;
  - its trace: 1 when its sensor last kicked it, fading with time (TRACE).
and one rule: a taste changes b by RATE x taste x trace (bad tastes negative, scaled so b stays between 0 and 1).
The three factors: pre (the trace: this input was just active), post (a meal: it happens only after approach, so the
taste arriving is itself the sign that approach won), and the taste.

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

import paths

SIZE = 49
SENSE = 0.1         # sensor kicks per unit time (at full strength)
DECIDE = 3          # choice: pulses an output must receive first to win
DELAY = 10.0        # from a meal to its taste (the body's)
TRACE = 10.0        # how fast an input's trace fades (per unit time; must outlast DELAY, not the sheet)
RATE = 1.0          # how far one taste moves the balance
INNATE = 0.75       # the balance an input cell is born with (above 0.5: drawn to anything it sees)
GAP = 50.0          # choice: mean time between one block's pulses arriving and the next block
STEP = 1.0          # line: how far one pulse at an output moves the robot
REACH = 8.0         # line: how far a block's colour carries
LOST = 24.0         # line: a block further than this is gone
SPAWN = 0.01        # line: new blocks per unit time (at most MOST in the world)
MOST = 2
CONTRAST = True     # a taste moves an input's balance by how far its trace exceeds its neighbours' (else by its trace)
FLAVOUR = True      # line: a block being eaten kicks its colour's sensor (the food in the mouth, sensed strongly)
COLOURS = ("red", "blue")
OUTPUTS = ("approach", "avoid")


def places(n: int) -> dict:
    """Where the sensors' input cells and the outputs are: the two inputs side by side by the west edge (a sensor
    organ), outputs by the east, as fractions of the sheet."""
    at = lambda fy, fx: (int(round(fy * (n - 1))), int(round(fx * (n - 1))))
    r, c = at(0.5, 0.1)
    return {"red": (r, c), "blue": (r + 1, c), "approach": at(0.35, 0.9), "avoid": at(0.65, 0.9)}


class Brain:
    """The sheet: two path channels, and the two input cells' balances and traces. The organs (where the inputs and
    outputs are) are mounted at birth, or later (mount), once the sheet has named them; until then the channels have
    no ends and nothing happens in them."""

    def __init__(self, seed: int, dt: float, where: dict | None = None):
        paths.SIZE = SIZE
        paths.KICK_SENDS = False                    # pulses only when a sensor says
        self.dt = dt
        self.channels = [paths.born(seed + 101 * (i + 1)) for i in range(2)]
        self.none = paths.masks(())
        self.mount(places(SIZE) if where is None else where)
        self.rng = np.random.default_rng(seed + 7)
        self.balance = {c: INNATE + 0.02 * self.rng.standard_normal() for c in COLOURS}
        self.kicked = {c: -1e9 for c in COLOURS}    # when each input's sensor last kicked it
        self.got = [0.0, 0.0]
        self.t = 0.0

    def mount(self, where: dict) -> None:
        """The organs: cells for "red", "blue", "approach" and "avoid" (all four, or none)."""
        self.where = where
        self.sources = paths.masks(tuple(where[c] for c in COLOURS if c in where))
        self.sinks = [paths.masks((where[o],) if o in where else ()) for o in OUTPUTS]

    def tick(self, kicks=()) -> list:
        """One tick: each kicked input cell sends a pulse one way or the other, by its balance. Returns how many
        pulses reached each output in this tick."""
        send = [self.none.copy(), self.none.copy()]
        for c in kicks:
            self.kicked[c] = self.t
            send[0 if self.rng.random() < self.balance[c] else 1][self.where[c]] = True
        for i, (s, s2, tag, k) in enumerate(self.channels):
            paths._ticks(s, s2, tag, self.sources, self.sinks[i], self.none, self.none, send[i], 1, self.dt, k)
        self.t += self.dt
        if not self.where:
            return [0.0, 0.0]
        got = [self.channels[i][0][paths.ARRIVED][self.where[o]] for i, o in enumerate(OUTPUTS)]
        new = [g - h for g, h in zip(got, self.got)]
        self.got = got
        return new

    def taste(self, taste: float) -> None:
        """A taste reaches both input cells: each moves its balance by RATE x taste x its trace (or, with CONTRAST,
        by how far its trace exceeds its neighbour's: the inputs are side by side)."""
        traces = {c: np.exp(-(self.t - self.kicked[c]) / TRACE) for c in COLOURS}
        for c in COLOURS:
            trace = traces[c]
            if CONTRAST:
                trace = max(0.0, trace - max(traces[o] for o in COLOURS if o != c))
            b = self.balance[c]
            self.balance[c] = b + RATE * taste * trace * (b if taste < 0 else 1.0 - b)

    def flying(self) -> bool:
        return any((ch[0][paths.FIRE_LEFT] > 0).any() for ch in self.channels)


def choice(seed: int = 0, dt: float = 1.0, poison: str = "red"):
    """The choice world, without end: yields (brain, the block in view or None, events) after every tick; an event
    is (colour, the output that won, the balance before, time)."""
    brain, rng = Brain(seed, dt), np.random.default_rng(seed)
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
    brain, rng = Brain(seed, dt), np.random.default_rng(seed)
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
            s = brain.channels[i][0]
            shade = np.where(s[paths.PATH] > 0.5, np.maximum(shade, level), shade)
        for s, *_ in brain.channels:
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
