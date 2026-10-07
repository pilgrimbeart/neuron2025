"""Learning in place, by a three-factor rule: on a bare sheet, a red and a blue sensor are wired by grown paths to an
APPROACH and an AVOID output, and the sheet learns, from tastes in random order, which colour is poison.

    python learn.py [--size N] [--seed N] [--dt X] [--blocks N] [--poison red|blue] [--watch]

The sheet: two breadcrumb channels (paths.py), one per output. In each, both sensors are sources and the output the
sink, so four routes grow (red and blue, to approach and to avoid); a channel per target, so they cross freely.
The input cell (where the paths start, and where taste arrives, as in the robot) holds two numbers:
  - its balance b: each pulse it sends goes towards APPROACH with chance b, else towards AVOID;
  - its trace: 1 when its sensor last kicked it, fading with time (TRACE).
and one rule: a taste changes b by RATE x taste x trace (bad tastes negative, scaled so b stays between 0 and 1).
The three factors: pre (the trace: this input was just active), post (a meal: it happens only after approach, so the
taste arriving is itself the sign that approach won), and the taste.
The body and world (not the sheet): a block appears, red or blue at random; while it is in view its colour's sensor
kicks its input cell at random times; the first output to receive DECIDE pulses wins. Approach: the block is eaten,
and DELAY later tasted at both input cells, bad if it is the poison colour. Avoid: it goes, untasted. The next block
comes a random while after the pulses in flight have arrived. Nothing waits a fixed time that depends on the size.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import paths

SIZE = 49
SENSE = 0.1         # sensor kicks per unit time while a block is in view
DECIDE = 3          # pulses an output must receive first to win
DELAY = 10.0        # from a meal to its taste (the body's)
TRACE = 50.0        # how fast an input's trace fades (per unit time; must outlast DELAY, not the sheet)
RATE = 0.6          # how far one taste moves the balance
GAP = 50.0          # mean time between one block's pulses arriving and the next block
COLOURS = ("red", "blue")
OUTPUTS = ("approach", "avoid")


def places(n: int) -> dict:
    """Where the sensors' input cells and the outputs are: inputs by the west edge, outputs by the east, as
    fractions of the sheet."""
    at = lambda fy, fx: (int(round(fy * (n - 1))), int(round(fx * (n - 1))))
    return {"red": at(0.35, 0.1), "blue": at(0.65, 0.1), "approach": at(0.35, 0.9), "avoid": at(0.65, 0.9)}


def life(seed: int = 0, dt: float = 1.0, poison: str = "red"):
    """One life, without end: yields after every tick (time, the two channels' numbers, the inputs' balances, the
    current block's colour or None, and a list of what happened at each block so far: (colour, the output that won,
    the balance before))."""
    paths.SIZE = SIZE
    paths.KICK_SENDS = False                        # pulses only when a sensor says
    where = places(SIZE)
    sources = paths.masks((where["red"], where["blue"]))
    channels = [paths.born(seed + 101 * (i + 1)) for i in range(2)]
    sinks = [paths.masks((where[o],)) for o in OUTPUTS]
    none = paths.masks(())
    rng = np.random.default_rng(seed)
    balance = {c: 0.5 + 0.02 * rng.standard_normal() for c in COLOURS}
    kicked = {c: -1e9 for c in COLOURS}             # when each input's sensor last kicked it
    block, shown, arrived, tastes, events = None, 0.0, None, [], []
    wait = rng.exponential(GAP)
    t = 0.0
    while True:
        send = [none.copy(), none.copy()]
        if block is not None and rng.random() < SENSE * dt:     # the sensor kicks its input cell, which sends a
            kicked[block] = t                                   # pulse one way or the other, by its balance
            send[0 if rng.random() < balance[block] else 1][where[block]] = True
        for i, (s, s2, tag, k) in enumerate(channels):
            paths._ticks(s, s2, tag, sources, sinks[i], none, none, send[i], 1, dt, k)
        t += dt
        got = [channels[i][0][paths.ARRIVED][where[o]] for i, o in enumerate(OUTPUTS)]
        if block is not None:
            counts = [g - a for g, a in zip(got, arrived)]
            if max(counts) >= DECIDE:                           # the body acts on the first output to DECIDE
                won = OUTPUTS[int(np.argmax(counts))]
                events.append((block, won, balance[block], t))
                if won == "approach":
                    tastes.append((t + DELAY, -1.0 if block == poison else 1.0))
                block, wait = None, rng.exponential(GAP)
        else:
            flying = any((ch[0][paths.FIRE_LEFT] > 0).any() for ch in channels)
            if not flying:
                wait -= dt
                if wait <= 0:
                    block, shown, arrived = COLOURS[rng.integers(2)], t, got
        for due, taste in [x for x in tastes if x[0] <= t]:     # a taste reaches both input cells
            tastes.remove((due, taste))
            for c in COLOURS:
                trace = np.exp(-(t - kicked[c]) / TRACE)
                b = balance[c]
                balance[c] = b + RATE * taste * trace * (b if taste < 0 else 1.0 - b)
        yield t, channels, balance, block, events


def run(seed: int = 0, dt: float = 1.0, poison: str = "red", blocks: int = 30):
    """A life until `blocks` blocks have been met. Returns the events and the final balances."""
    for t, channels, balance, block, events in life(seed, dt, poison):
        if len(events) >= blocks:
            return events, dict(balance), t


def summary(events, poison: str) -> str:
    """How often each colour was approached in the first and second half of the blocks."""
    half = len(events) // 2
    out = []
    for name, part in (("first half", events[:half]), ("second half", events[half:])):
        rates = []
        for c in COLOURS:
            seen = [e for e in part if e[0] == c]
            got = sum(e[1] == "approach" for e in seen)
            rates.append(f"{c}{' (poison)' if c == poison else ''} {got}/{len(seen)}")
        out.append(f"{name}: approached " + ", ".join(rates))
    bites = [i for i, e in enumerate(events) if e[0] == poison and e[1] == "approach"]
    out.append(f"poison eaten at blocks {bites}")
    return "; ".join(out)


def watch(seed: int = 0, dt: float = 1.0, poison: str = "red") -> None:
    """A life, live, in greys: the approach channel's path light grey, the avoid channel's mid grey, pulses white;
    the inputs lettered R and B (with their balances), the outputs A and V. Keys: r new seed (and poison colour);
    space pauses; up and down arrows change the speed; Escape quits."""
    import pygame
    pygame.init()
    n = SIZE
    scale = max(1, 640 // n)
    side = n * scale
    screen = pygame.display.set_mode((side, side + 72))
    pygame.display.set_caption("learn: which colour is poison")
    font = pygame.font.SysFont(None, 22)
    where = places(n)
    ticks_per_frame, paused = 2, False
    clock = pygame.time.Clock()
    lives = life(seed, dt, poison)
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
                    lives = life(seed, dt, poison)
        if paused:
            clock.tick(30)
            continue
        started, done = time.time(), 0
        while done < ticks_per_frame and (done == 0 or time.time() - started < 0.2):
            t, channels, balance, block, events = next(lives)
            done += 1
        shade = np.zeros((n, n))
        for i, level in ((1, 0.4), (0, 0.7)):
            s = channels[i][0]
            shade = np.where(s[paths.PATH] > 0.5, np.maximum(shade, level), shade)
        for s, *_ in channels:
            shade = np.where((s[paths.PATH] > 0.5) & (s[paths.FIRE_LEFT] > 0), 1.0, shade)
        grey = (255 * shade).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        for name, letter in (("red", "R"), ("blue", "B"), ("approach", "A"), ("avoid", "V")):
            r, c = where[name]
            text = letter + (f" {balance[name]:.2f}" if name in balance else "")
            framed = name == block
            screen.blit(font.render(text, True, (0, 0, 0) if framed else (255, 255, 255),
                                    (255, 255, 255) if framed else (70, 70, 70)), (c * scale - 4, r * scale - 4))
        last = events[-6:]
        history = "  ".join(f"{e[0][0].upper()}:{'eat' if e[1] == 'approach' else 'avoid'}" for e in last)
        status = f"seed {seed}   time {t:6.0f}   poison {poison}   blocks {len(events)}   in view: {block or '-'}"
        screen.blit(font.render(status, True, (255, 255, 255)), (6, side + 6))
        screen.blit(font.render("recent: " + history, True, (255, 255, 255)), (6, side + 28))
        screen.blit(font.render("R/B balance = chance a pulse goes to A (approach), else V (avoid)   "
                                "r new seed   space   up/down   Esc", True, (255, 255, 255)), (6, side + 50))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--poison", choices=COLOURS, default="red")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    SIZE = args.size
    if args.watch:
        watch(args.seed, args.dt, args.poison)
        raise SystemExit
    events, balance, t = run(args.seed, args.dt, args.poison, args.blocks)
    print(summary(events, args.poison))
    print(f"balances at the end: " + ", ".join(f"{c} {b:.2f}" for c, b in balance.items()) + f"; time {t:.0f}")
