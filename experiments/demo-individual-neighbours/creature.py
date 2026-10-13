"""A new-born creature whose brain builds itself, then learns which colour is food. The body, the world and the views;
everything the brain's cells are and do is in cell.py.

    python creature.py                  # watch a life (r: new life, space: pause, up/down: speed, Esc: quit)
    python creature.py --measure 16     # 16 lives with learning and 16 without: how much poison it eats
    python creature.py --check          # every cell is born the same, bar noise

The body touches the brain only at cells by the names they took (R, B, A, V), cell by cell: it kicks the cell named
R when its red sensor fires (B: blue), delivers tastes to R and B, and fires a thruster for each pulse arriving at a
cell named A (or V). Nothing else goes in or out.

The body: a square. R and B at the head see blocks of their colour in the corridor its mouth will sweep (fading with
distance). A at the tail pushes it forward; V, at a front corner, pushes it back and turns it, and also twitches by
itself now and then (a tumble): run and tumble. A block the head touches is eaten: its colour's sensor fires once
(the flavour), the creature chews with its eyes shut, then the taste arrives, bad if it was poison. The world: a
torus of red and blue blocks; an eaten block reappears elsewhere.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import cell

SIZE = 49           # the brain: SIZE x SIZE cells
BODY = 4.0          # the body's side, in world units
WORLD = 60.0        # the world's side (a torus)
BLOCKS = 18         # blocks of each colour
EAT = 1.0           # a block this near the front edge (the mouth: the whole head end) is eaten
SENSE = 0.21        # sensor kicks per tick at full strength
REACH = 10.0        # how far a block's colour carries
PUSH = 0.7          # a thruster firing adds this much speed
V_BACK = 0.5        # V pushes back this much as hard as A pushes forward (it is angled: the rest turns the body)
TURN = 0.05         # V firing adds this much turning speed (radians per tick)
DRAG = 0.2          # speed and turning fade at this rate
TUMBLE = 0.02       # V twitches by itself this often (per tick): a tumble, so it searches when it sees nothing
CHEW = 17           # ticks from a meal to its taste, eyes shut (news of the flavour crosses from R to B meanwhile)
NEWS_GLOW = 20      # watching: news younger than this glows
COLOURS = ("red", "blue")
SENSOR = {"red": cell.R, "blue": cell.B}


class Body:
    """Where the creature is and how it moves; the world's blocks."""

    def __init__(self, rng):
        self.rng = rng
        self.pos = np.array([WORLD / 2, WORLD / 2])
        self.heading = rng.uniform(0, 2 * math.pi)
        self.speed, self.spin = np.zeros(2), 0.0
        self.blocks = [[*self.somewhere(), c] for c in COLOURS for _ in range(BLOCKS)]

    def somewhere(self):
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
        """How strongly the head's sensor sees this colour: blocks in the corridor ahead, nearer ones more; at most 1."""
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
            self.speed += PUSH * self.forward()
        else:
            self.speed -= V_BACK * PUSH * self.forward()
            self.spin -= TURN

    def move(self) -> list:
        """Moves on a tick; returns the colours of blocks eaten (each reappears elsewhere)."""
        self.pos = (self.pos + self.speed) % WORLD
        self.heading += self.spin
        self.speed *= math.exp(-DRAG)
        self.spin *= math.exp(-DRAG)
        eaten, f = [], self.forward()
        for b in self.blocks:
            d = self.wrap(np.array(b[:2]) - self.pos)
            if 0 < d @ f < BODY / 2 + EAT and abs(d @ np.array([f[1], -f[0]])) < BODY / 2 + EAT / 2:
                eaten.append(b[2])
                b[0], b[1] = self.somewhere()
        return eaten


class Life:
    """A brain and a body in a world. step() runs one tick."""

    def __init__(self, seed: int = 0, poison: str = "red", learning: bool = True):
        self.rng = np.random.default_rng(seed)
        self.poison, self.learning = poison, learning
        self.s = cell.born(SIZE, self.rng)
        self.s2 = np.empty_like(self.s)
        cell.seed(seed)
        self.body = Body(self.rng)
        self.t = 0
        self.meals, self.firings = [], []                   # (tick, colour), (tick, "A" or "V")
        self.chewing = None                                 # (when the taste is due, the colour eaten)
        self.flavour = None                                 # the colour just eaten: its sensor fires once
        self.yuck = -1e9                                    # when a bad taste last arrived
        self.arrived = np.zeros((2, SIZE, SIZE))
        self.moving = None                                  # when the first pulse reached a thruster

    def step(self) -> None:
        name = self.s[cell.NAME]
        kick = np.zeros((SIZE, SIZE), np.bool_)
        taste = np.zeros((SIZE, SIZE))
        if self.flavour is not None:
            kick |= name == SENSOR[self.flavour]
            self.flavour = None
        if self.chewing is None:                            # eyes open: sensors fire at what they see
            for c in COLOURS:
                if self.rng.random() < SENSE * self.body.sees(c):
                    kick |= name == SENSOR[c]
        elif self.chewing[0] <= self.t:                     # done chewing: the taste arrives at R and B
            bad = self.chewing[1] == self.poison
            taste[(name == cell.R) | (name == cell.B)] = -1.0 if bad else 1.0
            if bad:
                self.yuck = self.t
            self.chewing = None
        cell.tick(self.s, self.s2, kick, taste, self.learning)
        arrived = self.s[[cell.CHANNEL[0] + cell.ARRIVED, cell.CHANNEL[1] + cell.ARRIVED]]
        new, self.arrived = arrived - self.arrived, arrived
        fired = ["A"] * int(new[0][name == cell.A].sum()) + ["V"] * int(new[1][name == cell.V].sum())
        if fired and self.moving is None:
            self.moving = self.t
        if (name == cell.V).any() and self.rng.random() < TUMBLE:
            fired.append("V")                               # a tumble
        for f in fired:
            self.body.fire(f)
            self.firings.append((self.t, f))
        for c in self.body.move():
            self.meals.append((self.t, c))
            if self.chewing is None:
                self.chewing, self.flavour = (self.t + CHEW, c), c
        self.t += 1

    def balance(self, colour: str) -> float:
        """The balance of the cell named for this colour's sensor (watching only)."""
        cells = self.s[cell.NAME] == SENSOR[colour]
        return float(self.s[cell.BALANCE][cells].mean()) if cells.any() else cell.INNATE


def one_life(job):
    """(seed, poison, learning, ticks) -> poison's share of meals after moving began, by thirds: [(poison, food)]."""
    seed, poison, learning, ticks = job
    life = Life(seed, poison, learning)
    while life.t < ticks:
        life.step()
    start = life.moving or ticks
    out = []
    for i in range(3):
        a, b = start + i * (ticks - start) / 3, start + (i + 1) * (ticks - start) / 3
        part = [c for m, c in life.meals if a <= m < b]
        out.append((sum(c == poison for c in part), sum(c != poison for c in part)))
    return out, life.balance(poison), life.balance("blue" if poison == "red" else "red")


def measure(lives: int, ticks: int = 8000) -> None:
    """Half the lives with red as poison, half blue; with learning and without."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    for learning in (True, False):
        jobs = [(i // 2, COLOURS[i % 2], learning, ticks) for i in range(lives)]
        with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(one_life, jobs))
        thirds = [[sum(r[0][i][k] for r in results) for k in (0, 1)] for i in range(3)]
        share = "  ".join(f"{100 * p / max(1, p + f):3.0f}% of {p + f:3d}" for p, f in thirds)
        print(f"learning {'on ' if learning else 'off'}: poison's share of meals in each third of life: {share};"
              f"  balance at the end: poison {np.mean([r[1] for r in results]):.2f},"
              f" food {np.mean([r[2] for r in results]):.2f}")


def check(seed: int = 0) -> bool:
    """Every cell is born the same, bar the layers that are random by design."""
    s = cell.born(SIZE, np.random.default_rng(seed))
    differ = [i for i in range(cell.LAYERS) if np.ptp(s[i]) > 0 and i not in cell.NOISE_LAYERS]
    print(f"{cell.LAYERS} numbers per cell; random at birth: tag, x, y, balance; any others differing: {differ or 'none'}")
    return not differ


def watch(seed: int = 0) -> None:
    """A life, live. Left, the brain, in greys: the frame faintly, routes to A light grey and to V mid grey, pulses
    white, organs outlined and lettered; news of a kick spreading from R or B as a fading red glow (R and B with their balances). Right, the world: blocks lettered R
    (dark) and B (light), the creature a square with its head marked."""
    import pygame
    pygame.init()
    scale = max(1, 490 // SIZE)
    side, wpx = SIZE * scale, 490
    screen = pygame.display.set_mode((side + 12 + wpx, max(side, wpx) + 72))
    pygame.display.set_caption("a brain builds itself, then learns which colour is food")
    font, small = pygame.font.SysFont(None, 22), pygame.font.SysFont(None, 18)
    per_frame, paused, clock = 2, False, pygame.time.Clock()
    poison = COLOURS[seed % 2]
    life = Life(seed, poison)
    to_px = lambda p: (side + 12 + int(p[0] / WORLD * wpx), int((WORLD - p[1]) / WORLD * wpx))
    letters = {cell.R: "R", cell.B: "B", cell.A: "A", cell.V: "V"}
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    per_frame *= 2
                elif event.key == pygame.K_DOWN:
                    per_frame = max(1, per_frame // 2)
                elif event.key == pygame.K_r:
                    seed += 1
                    poison = COLOURS[seed % 2]
                    life = Life(seed, poison)
        if not paused:
            started = time.time()
            for i in range(per_frame):
                life.step()
                if time.time() - started > 0.2:
                    break
        s = life.s
        screen.fill((0, 0, 0))
        bands = (np.floor(8 * np.clip(s[cell.X], 0, 0.999)) + np.floor(8 * np.clip(s[cell.Y], 0, 0.999))) % 2
        shade = 0.06 + 0.1 * bands
        for ch, level in ((1, 0.45), (0, 0.75)):
            shade = np.where(s[cell.CHANNEL[ch] + cell.PATH] > 0.5, np.maximum(shade, level), shade)
        for c in cell.CHANNEL:
            shade = np.where((s[c + cell.PATH] > 0.5) & (s[c + cell.PULSE] > 0), 1.0, shade)
        glow = np.clip(1.0 - s[cell.NEWS] / NEWS_GLOW, 0.0, 1.0)          # news of a kick: a fading red glow
        rgb = np.stack([np.maximum(shade, 0.8 * glow), shade * (1 - 0.6 * glow), shade * (1 - 0.6 * glow)], axis=-1)
        rgb = (255 * rgb).astype(np.uint8).transpose(1, 0, 2).repeat(scale, 0).repeat(scale, 1)
        screen.blit(pygame.surfarray.make_surface(rgb), (0, 0))
        yuck = life.t - life.yuck < 60
        for r, c in np.argwhere(s[cell.NAME] > 0):
            k = int(s[cell.NAME, r, c])
            text = letters[k]
            if k in (cell.R, cell.B):
                text += f" {s[cell.BALANCE, r, c]:.2f}"
            lit = yuck and k == SENSOR[poison]
            label = font.render(text, True, (0, 0, 0) if lit else (255, 255, 255), (255, 255, 255) if lit else (60, 60, 60))
            above = r * scale - 20 if r * scale > 24 else r * scale + scale + 2      # beside the cell, not over it
            screen.blit(label, (min(max(0, c * scale - 4), side - label.get_width()), above))
            pygame.draw.rect(screen, (255, 255, 255), (c * scale, r * scale, scale, scale), 2)
        pygame.draw.rect(screen, (25, 25, 25), (side + 12, 0, wpx, wpx))
        body = life.body
        for x, y, c in body.blocks:
            px = to_px((x, y))
            pygame.draw.rect(screen, (90, 90, 90) if c == "red" else (210, 210, 210), (px[0] - 8, px[1] - 8, 16, 16))
            screen.blit(small.render(c[0].upper(), True, (255, 255, 255) if c == "red" else (0, 0, 0)), (px[0] - 4, px[1] - 6))
        f = body.forward()
        r = np.array([f[1], -f[0]])
        half = BODY / 2
        corners = [body.pos + half * (f + r), body.pos + half * (f - r), body.pos + half * (-f - r), body.pos + half * (-f + r)]
        pygame.draw.polygon(screen, (255, 255, 255) if yuck else (160, 160, 160), [to_px(p) for p in corners], 5 if yuck else 2)
        pygame.draw.circle(screen, (255, 255, 255), to_px(body.head()), 4)
        recent = {k for when, k in life.firings[-4:] if life.t - when < 6}
        if "A" in recent:
            pygame.draw.circle(screen, (255, 255, 255), to_px(body.pos - half * f), 6, 2)
        if "V" in recent:
            pygame.draw.circle(screen, (255, 255, 255), to_px(body.pos + half * (0.6 * f + 0.6 * r)), 6, 2)
        if yuck:
            px = to_px(body.pos)
            screen.blit(font.render("yuck!", True, (0, 0, 0), (255, 255, 255)), (px[0] + 14, px[1] - 28))
        elif life.chewing:
            px = to_px(body.pos)
            screen.blit(small.render("chewing", True, (255, 255, 255)), (px[0] + 14, px[1] - 24))
        named = set(np.unique(s[cell.NAME]).astype(int)) - {0}
        stage = ("where am I" if len(named) < 4 else "wiring" if life.moving is None else "moving and learning")
        eaten = {c: sum(1 for _, m in life.meals if m == c) for c in COLOURS}
        base = max(side, wpx)
        lines = (f"tick {life.t:6d}   stage: {stage}   poison: {poison}   eaten: red {eaten['red']}, blue {eaten['blue']}",
                 "recent meals: " + " ".join(m[0].upper() for _, m in life.meals[-14:]),
                 f"{per_frame} ticks per frame   r new life   space pause   up/down speed   Esc quit")
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (255, 255, 255)), (6, base + 6 + 22 * i))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measure", type=int, metavar="LIVES", help="measure over this many lives (headless)")
    parser.add_argument("--ticks", type=int, default=8000, help="ticks per life when measuring")
    parser.add_argument("--check", action="store_true", help="check every cell is born the same, bar noise")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check(args.seed) else 1)
    if args.measure:
        measure(args.measure, args.ticks)
    else:
        watch(args.seed)
