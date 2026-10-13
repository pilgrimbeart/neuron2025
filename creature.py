"""A new-born creature: its brain assembles itself in stages, then it moves, and learns which colour is food.

    python creature.py [--size N] [--seed N] [--poison red|blue] [--time T] [--watch] [--check]

The brain is a square sheet of identical cells (each updating at random times), developing in stages, with no timer
between them: each starts when the one before has produced what it needs.
  1. Where am I (frame.py): the corners elect an origin, the edges are labelled, every cell comes to hold x and y
     (0 to 1), and cells name themselves organs by them: R at the middle of the north edge, A at the middle of the
     south edge, V at north-east (0.8, 0.8), and B 6 cells east of R (3 either side of the north middle: each cell
     knows how wide a cell is, 1 / the width, from its own counts), so the news between them can be watched.
  2. Wiring (paths.py): a cell named R or B is a source, A or V a sink, and two breadcrumb channels grow routes from
     R and B to A and to V. The body's sensors and thrusters are plugged into whichever cells took those names (the
     head is wherever the sheet put north).
  3. Behaviour and learning (learn.py's learning layer): a sensor kick makes its input cell send a pulse towards A
     with chance its balance b (born 0.9: drawn to anything it sees), else towards V; news of a kick spreads to the
     other input, wiping its trace; a taste moves b by the cell's trace (three-factor learning); a bad taste makes
     the cell send towards V for a while (DISGUST): the creature recoils.
Every cell holds all of these layers and the same rule; a cell's layers read each other (named R, it is a source),
and the body touches only cells by the names they have taken (python creature.py --check: born identical bar noise).
The body (not the sheet): a square; R and B at the head, facing forward, see blocks of their colour in the corridor
the mouth will sweep (ahead, within its width, fading with distance); A at the tail pushes it forward; V, at the front
corner, pushes it back and turns it (a push off its centre line). V's thruster also twitches now and then by itself
(a spontaneous tumble), so the creature searches: run and tumble. A block touched by the head end (the mouth) is eaten:
its colour's sensor kicks once (the flavour in the mouth); its eyes are shut while it chews, and then the taste
arrives at R and B, bad if it is poison. The body is gentle (TEMPO), as pulses take long to cross the brain. The world: a torus with blocks of both
colours; an eaten block reappears elsewhere.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import frame
import learn
import paths

SIZE = 49
ORGANS = {"R": (0.5, 1.0, -3, 0), "B": (0.5, 1.0, 3, 0), "A": (0.5, 0.0), "V": (0.8, 0.8)}   # where the sheet
                    # names its organs: x, y as fractions of the body, plus an offset in cells (R and B 6 apart)
BODY = 4.0          # the body's side, in world units
WORLD = 60.0        # the world's side (a torus)
BLOCKS = 18         # blocks of each colour
EAT = 1.0           # a block this near the front edge (the mouth: the whole head end) is eaten
SENSE = 0.3         # sensor kicks per unit time at full strength
REACH = 10.0        # how far a block's colour carries
PUSH = 1.0          # a thruster firing adds this much speed
V_BACK = 0.5        # V pushes back this much as hard as A pushes forward (it is angled: the rest turns the body)
TURN = 0.07         # V firing adds this much turning speed (radians per unit time)
DRAG = 0.2          # speed and turning fade at this rate
TUMBLE = 0.005      # the V cell fires by itself this often (per unit time)
DISGUST = 40.0      # a bad taste makes R and B send towards V this long (at the sensors' rate): the creature recoils
DELAY = 12.0        # from a meal to its taste (chewing); its eyes are shut meanwhile, so nothing else is seen, and
                    # the news of the meal's flavour kick reaches the other input before the taste does
TRACE = 30.0        # how long an input's trace lasts (it must outlast the chewing)
TEMPO = 0.7         # how strong and quick the body is (sensing, pushes, turns, tumbles): gentle, as pulses take long
                    # to cross the brain and several are in flight at once (a bigger brain needs a gentler body)
INNATE = 0.9        # the input cells' born balance: drawn to anything seen (stray turns spoil a slow brain's runs)
COLOURS = learn.COLOURS


def organs_of(s) -> dict | None:
    """For watching only (the creature never uses it): the cells that have named themselves organs, once each name is
    held by exactly one cell: {"R", "B", "A", "V"} -> (row, column). None until then."""
    out = {}
    for i, name in enumerate(ORGANS):
        cells = np.argwhere(s[frame.NAME] == i + 1)
        if len(cells) != 1:
            return None
        out[name] = tuple(int(v) for v in cells[0])
    return out


class Body:
    """Where the creature is and how it moves; the world's blocks."""

    def __init__(self, rng):
        self.rng = rng
        self.pos = np.array([WORLD / 2, WORLD / 2])
        self.heading = rng.uniform(0, 2 * math.pi)
        self.speed, self.spin = np.zeros(2), 0.0
        self.blocks = [[*self.somewhere(), c] for c in COLOURS for _ in range(BLOCKS)]
        v = ORGANS["V"]
        self.v_side = 1.0 if v[0] > 0.5 else -1.0      # V east of the centre line: backing turns it clockwise

    def somewhere(self):
        """A random place not too near the creature."""
        while True:
            p = self.rng.uniform(0, WORLD, 2)
            if np.linalg.norm(self.wrap(p - self.pos)) > 3 * EAT + BODY:
                return p

    @staticmethod
    def wrap(d):
        return (d + WORLD / 2) % WORLD - WORLD / 2

    def forward(self):
        return np.array([math.cos(self.heading), math.sin(self.heading)])

    def head(self):
        return self.pos + self.forward() * BODY / 2

    def sees(self, colour: str) -> float:
        """How strongly the head's sensor for this colour sees: blocks in the corridor the mouth will sweep (ahead,
        within its width), fading with distance; at most 1."""
        f, h, seen = self.forward(), self.head(), 0.0
        side = np.array([f[1], -f[0]])
        for x, y, c in self.blocks:
            if c == colour:
                d = self.wrap(np.array([x, y]) - h)
                if d @ f > 0 and abs(d @ side) < BODY / 2 + EAT / 2:
                    seen += 1 / (1 + (np.linalg.norm(d) / REACH) ** 2)
        return min(1.0, seen)

    def fire(self, thruster: str) -> None:
        if thruster == "A":
            self.speed += TEMPO * PUSH * self.forward()
        else:
            self.speed -= TEMPO * V_BACK * PUSH * self.forward()
            self.spin -= TEMPO * TURN * self.v_side

    def move(self, dt: float) -> list:
        """Moves on; returns the colours of blocks eaten (each reappears elsewhere)."""
        self.pos = (self.pos + self.speed * dt) % WORLD
        self.heading += self.spin * dt
        self.speed *= math.exp(-DRAG * dt)
        self.spin *= math.exp(-DRAG * dt)
        eaten = []
        f = self.forward()
        for b in self.blocks:
            d = self.wrap(np.array(b[:2]) - self.pos)
            ahead, aside = d @ f, abs(d @ np.array([f[1], -f[0]]))
            if 0 < ahead < BODY / 2 + EAT and aside < BODY / 2 + EAT / 2:
                eaten.append(b[2])
                b[0], b[1] = self.somewhere()
        return eaten


def life(seed: int = 0, dt: float = 1.0, poison: str = "red"):
    """One life, without end: yields (time, the frame's numbers, the brain, the body, the stage, meals [(time,
    colour)], thruster firings [(time, "A" or "V")]) after every tick.

    The sheet is every cell's own numbers in layers (frame, learning, two path channels), and each cell's layers read
    each other: a cell is a source in both channels when it has named itself R or B, a sink in the approach channel
    when A, in the avoid channel when V. The body touches only cells by the names they have taken: it kicks the cells
    named R (B) when the red (blue) sensor fires, delivers tastes to the cells named R and B, and pushes A's (V's)
    thruster for every pulse arriving at a cell named A (V). The stage is only watched, never used."""
    frame.SIZE = learn.SIZE = SIZE
    frame.PLACES = ORGANS
    learn.INNATE, learn.DISGUST, learn.RECOIL, learn.TRACE = INNATE, DISGUST, TEMPO * SENSE, TRACE
    rng = np.random.default_rng(seed)
    body = Body(rng)
    grow = frame.life(seed)
    t, s, tag = next(grow)
    brain = learn.Brain(seed, dt, tag)                      # one tag per cell, for every layer
    id_ = {name: i + 1 for i, name in enumerate(ORGANS)}
    stage, meals, firings, tastes, flavour = "where am I", [], [], [], []
    while True:
        name = s[frame.NAME]
        named = {o: name == id_[o] for o in ORGANS}
        sources = named["R"] | named["B"]
        kick = np.zeros_like(sources)
        for c, o in zip(COLOURS, "RB"):
            if c in flavour or (not tastes and rng.random() < TEMPO * SENSE * body.sees(c) * dt):
                kick |= named[o]
        flavour = []
        taste = np.zeros(name.shape)
        for due, value in [x for x in tastes if x[0] <= t]:
            tastes.remove((due, value))
            taste[sources] += value
        got = brain.tick(sources, [named["A"], named["V"]], kick, taste)
        fired = ["A"] * int(got[0][named["A"]].sum()) + ["V"] * int(got[1][named["V"]].sum())
        if stage == "wiring" and fired:
            stage = "moving and learning"
        if named["V"].any() and rng.random() < TEMPO * TUMBLE * dt:
            fired.append("V")                                   # V's thruster twitches by itself: a tumble
        for f in fired:
            body.fire(f)
            firings.append((t, f))
        if stage == "where am I" and organs_of(s) is not None:
            stage = "wiring"
        for c in body.move(dt):
            meals.append((t, c))
            tastes.append((t + DELAY / TEMPO, -1.0 if c == poison else 1.0))
            flavour.append(c)                                   # the food in the mouth, sensed strongly
        yield t, s, brain, body, stage, meals, firings
        t, s, tag = next(grow)


def balances(s, brain) -> dict:
    """For watching: the balances of the cells named R and B (their mean, if a name is held by several)."""
    out = {}
    for c, i in zip(COLOURS, (1, 2)):
        cells = s[frame.NAME] == i
        out[c] = float(brain.l[learn.BALANCE][cells].mean()) if cells.any() else learn.INNATE
    return out


def check(seed: int = 0) -> bool:
    """Every cell is born the same, bar noise: at birth every layer of every cell holds the same number, except the
    layers that are random by design (each cell's tag, its frame's starting x and y, its balance). Prints each layer
    family and whether it passes."""
    frame.SIZE = learn.SIZE = SIZE
    s, tag = frame.born(seed)
    brain = learn.Brain(seed, 1.0, tag)
    noise = {("frame", frame.X), ("frame", frame.Y), ("learning", learn.BALANCE)}
    ok = True
    for family, layers in [("frame", s)] + [("learning", brain.l)] + [(f"path {i}", c) for i, (c, _, _) in
                                                                      enumerate(brain.channels)]:
        same = [i for i in range(layers.shape[0]) if (family, i) in noise or np.ptp(layers[i]) == 0.0]
        print(f"{family}: {len(same)} of {layers.shape[0]} layers the same everywhere or noise by design")
        ok &= len(same) == layers.shape[0]
    print(f"tag: one per cell, random; {'all cells identical bar noise' if ok else 'NOT IDENTICAL'}")
    return ok


def run(seed: int = 0, dt: float = 1.0, poison: str = "red", time_: float = 6000.0):
    """When each stage began, the meals of each colour in each third of the life after the first movement, and the
    balances at the end."""
    marks = {}
    for t, s, brain, body, stage, meals, firings in life(seed, dt, poison):
        marks.setdefault(stage, t)
        if t >= time_:
            break
    start = marks.get("moving and learning", time_)
    thirds = []
    for i in range(3):
        a, b = start + i * (time_ - start) / 3, start + (i + 1) * (time_ - start) / 3
        part = [c for m, c in meals if a <= m < b]
        thirds.append({c: part.count(c) for c in COLOURS})
    return marks, thirds, balances(s, brain)


def watch(seed: int = 0, dt: float = 1.0, poison: str = "red") -> None:
    """A life, live. Left, the brain: the frame (dim checkerboard of eighths of x and y), the routes to A (light grey)
    and to V (mid grey), pulses (white), news of a kick spreading from R or B (outlined cells), the organs lettered (R and B with
    their balances). Right, the world: blocks
    lettered R (dark) and B (light), the creature a square with its head marked, a thruster's side flashing as it
    fires. Keys: r new seed (and poison colour); space pauses; up and down arrows change the speed; Escape quits."""
    import pygame
    pygame.init()
    n = SIZE
    scale = max(1, 480 // n)
    side = n * scale
    wpx = 480
    screen = pygame.display.set_mode((side + 12 + wpx, max(side, wpx) + 72))
    pygame.display.set_caption("creature: a brain assembles itself, then learns which colour is food")
    font = pygame.font.SysFont(None, 22)
    small = pygame.font.SysFont(None, 18)
    ticks_per_frame, paused = 2, False                  # about 60 ticks a second: the whole story in two minutes
    clock = pygame.time.Clock()
    lives = life(seed, dt, poison)
    to_px = lambda p: (side + 12 + int(p[0] / WORLD * wpx), int((WORLD - p[1]) / WORLD * wpx))
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
            t, s, brain, body, stage, meals, firings = next(lives)
            done += 1
        organs = organs_of(s)
        balance = balances(s, brain)
        screen.fill((0, 0, 0))
        # the brain
        bands = (np.floor(8 * np.clip(s[frame.X], 0, 0.999)) + np.floor(8 * np.clip(s[frame.Y], 0, 0.999))) % 2
        shade = 0.06 + 0.1 * bands
        for i, level in ((1, 0.45), (0, 0.75)):
            p = brain.channels[i][0]
            shade = np.where(p[paths.PATH] > 0.5, np.maximum(shade, level), shade)
        for p, *_ in brain.channels:
            shade = np.where((p[paths.PATH] > 0.5) & (p[paths.FIRE_LEFT] > 0), 1.0, shade)
        grey = (255 * shade).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
        screen.blit(pygame.surfarray.make_surface(np.stack([grey] * 3, axis=-1)), (0, 0))
        for r, c in np.argwhere(brain.l[learn.NEWS] < learn.SHOW):  # news of a kick, spreading: outlined cells
            pygame.draw.rect(screen, (255, 255, 255), (c * scale, r * scale, scale, scale), 1)
        yuck = bool(meals) and meals[-1][1] == poison and t - meals[-1][0] < 60     # just ate poison
        if organs:
            labels = {"R": f"R {balance['red']:.2f}", "B": f"B {balance['blue']:.2f}", "A": "A", "V": "V"}
            for name, (r, c) in organs.items():
                lit = yuck and name == poison[0].upper()               # the poisoned colour's balance, falling
                screen.blit(font.render(labels[name], True, (0, 0, 0) if lit else (255, 255, 255),
                                        (255, 255, 255) if lit else (60, 60, 60)),
                            (min(max(0, c * scale - 4), side - 60), min(max(0, r * scale - 4), side - 18)))
        # the world
        pygame.draw.rect(screen, (25, 25, 25), (side + 12, 0, wpx, wpx))
        for x, y, c in body.blocks:
            px = to_px((x, y))
            fill = (90, 90, 90) if c == "red" else (210, 210, 210)
            pygame.draw.rect(screen, fill, (px[0] - 8, px[1] - 8, 16, 16))
            screen.blit(small.render(c[0].upper(), True, (255, 255, 255) if c == "red" else (0, 0, 0)), (px[0] - 4, px[1] - 6))
        f = body.forward()
        r = np.array([f[1], -f[0]])                             # the body's right
        half = BODY / 2
        corners = [body.pos + half * (f + r), body.pos + half * (f - r), body.pos + half * (-f - r), body.pos + half * (-f + r)]
        pygame.draw.polygon(screen, (255, 255, 255) if yuck else (160, 160, 160), [to_px(c) for c in corners],
                            5 if yuck else 2)
        if yuck:
            px = to_px(body.pos)
            screen.blit(font.render("yuck!", True, (0, 0, 0), (255, 255, 255)), (px[0] + 14, px[1] - 28))
        pygame.draw.circle(screen, (255, 255, 255), to_px(body.head()), 4)
        recent = {f_ for when, f_ in firings[-4:] if t - when < 6}
        if "A" in recent:
            pygame.draw.circle(screen, (255, 255, 255), to_px(body.pos - half * f), 6, 2)
        if "V" in recent:
            pygame.draw.circle(screen, (255, 255, 255), to_px(body.pos + half * (0.6 * f + 0.6 * body.v_side * r)), 6, 2)
        # status
        base = max(side, wpx)
        eaten = {c: sum(1 for _, m in meals if m == c) for c in COLOURS}
        last = [m[0].upper() for _, m in meals[-12:]]
        status = (f"seed {seed}   time {t:6.0f}   stage: {stage}   poison: {poison}   "
                  f"eaten red {eaten['red']}, blue {eaten['blue']}")
        screen.blit(font.render(status, True, (255, 255, 255)), (6, base + 6))
        screen.blit(font.render("recent meals: " + " ".join(last), True, (255, 255, 255)), (6, base + 28))
        screen.blit(font.render(f"{ticks_per_frame} ticks per frame   r new seed   space pause   up/down speed   Esc",
                                True, (255, 255, 255)), (6, base + 50))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--poison", choices=COLOURS, default="red")
    parser.add_argument("--time", type=float, default=6000.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--check", action="store_true", help="check every cell is born the same, bar noise")
    args = parser.parse_args()
    SIZE = args.size
    if args.check:
        raise SystemExit(0 if check(args.seed) else 1)
    if args.watch:
        watch(args.seed, args.dt, args.poison)
        raise SystemExit
    started = time.time()
    marks, thirds, balance = run(args.seed, args.dt, args.poison, args.time)
    print("stages began at: " + ", ".join(f"{k} {v:.0f}" for k, v in marks.items()))
    print("meals in each third after moving: " + "; ".join(", ".join(f"{c} {n}" for c, n in th.items()) for th in thirds))
    print("balances at the end: " + ", ".join(f"{c} {b:.2f}" for c, b in balance.items())
          + f"   (poison {args.poison}; {time.time() - started:.0f} s)")
