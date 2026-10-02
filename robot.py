"""A round robot run by a disc of identical cells (cell.py), in a walled arena of red and blue blocks.

  - sensors sit evenly round the rim, each a red and a blue input cell either side of an actuator cell; a sensor sees
    colour in the direction it faces (by cosine, fading with distance), and its input cell is kicked (v += KICK) at
    random, at RATE x what it sees per unit time (a Poisson process): sparse pulses, never levels
  - time: each tick is DT units of simulated time. Rates are per unit time and kicks, tastes and pushes are
    instantaneous, so behaviour shouldn't depend on DT (1 = the tick the rules were evolved at)
  - each actuator is a thruster: each time its cell's v rises through cell.FIRE_LEVEL, the robot is pushed PUSH away
    from that side. So approaching what one side sees takes the far side's thruster: signals must cross the disc
  - touching a block eats it, and it is tasted: the taste cell of the sensor group facing it (one cell in from that
    group's actuator, beside both its input cells) is kicked by TASTE_CONTACT, and the middle four cells of the disc
    by TASTE_CENTRE, in their own variable t: up (t += taste) for food, down (t -= taste) for poison. The block reappears elsewhere. (The two
    taste strengths are body genes: evolution can choose where taste lands.)
  - tasks, shortest first: move (no blocks), approach (one block, which is food), taste (one red block, which is food
    if red is food and poison otherwise), choose (one red, one blue, one of them food), forage (several of each),
    graze (several of each, all food, seen only near by: SHORT_REACH) and discriminate (as graze, one colour food)
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from numba import njit

import cell

TASKS = ("move", "approach", "taste", "choose", "forage", "graze", "discriminate")
RED, BLUE, BOTH = 0, 1, 2               # which colour is food (BOTH: both are)
SHORT_REACH = 4.0                       # how far colour carries in graze and discriminate: the nearest block dominates
PARTS = 3                               # lives are counted in parts (thirds), to see how they change


@dataclass(frozen=True)
class Body:
    grid: int = 12              # the disc of cells fits a grid x grid square
    sensors: int = 6            # each with a red and a blue input cell and an actuator between them


@dataclass(frozen=True)
class World:
    size: float = 30.0          # the arena is size x size, walled
    radius: float = 2.0         # the robot's radius
    reach: float = 8.0          # how far a block's colour carries
    rate: float = 0.05          # sensor pulses per unit time at full intensity
    push: float = 1.0           # how far one thruster firing moves the robot
    kick: float = 1.0           # how much a sensor pulse adds to v
    taste_contact: float = 1.0  # how much a taste adds to (or takes from) v where the block touched
    taste_centre: float = 0.0   # ... and in the middle of the disc
    dt: float = 1.0             # simulated time per tick

    def array(self) -> np.ndarray:
        return np.array([self.size, self.radius, self.reach, self.rate, self.push, self.kick, self.taste_contact,
                         self.taste_centre, self.dt])

    def for_task(self, task: str) -> "World":
        """This world as a task needs it: graze and discriminate see only near by."""
        return dataclasses.replace(self, reach=SHORT_REACH) if task in ("graze", "discriminate") else self

    def ticks(self, duration: float) -> int:
        """How many ticks make up a duration of simulated time."""
        return max(1, round(duration / self.dt))


class Geometry:
    """The disc of cells, and where the inputs and outputs sit on it. Cells are numbered; inputs, outputs and taste
    cells are cell numbers, and xy says where each cell is drawn."""

    def __init__(self, body: Body = Body()):
        n = body.grid
        c = (n - 1) / 2
        rim = n / 2 - 1
        self.size = n
        inside = lambda x, y: 0 <= x < n and 0 <= y < n and (x - c) ** 2 + (y - c) ** 2 <= (n / 2) ** 2
        self.xy = np.array([(x, y) for x in range(n) for y in range(n) if inside(x, y)])
        index = {tuple(p): i for i, p in enumerate(self.xy)}
        self.sheet = cell.Sheet([[index[(x + dx, y + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                  if (dx or dy) and inside(x + dx, y + dy)] for x, y in self.xy])
        at = lambda angle, r: index[(int(round(c + r * math.cos(angle))), int(round(c + r * math.sin(angle))))]
        s = body.sensors
        self.angles = np.array([2 * math.pi * i / s for i in range(s)])
        offset = 1.5 / rim
        self.inputs = np.array([[at(a - offset, rim), at(a + offset, rim)] for a in self.angles])    # (s, colour)
        self.outputs = np.array([at(a, rim) for a in self.angles])                                  # (s,)
        self.taste = np.array([at(a, rim - 1) for a in self.angles])                                # (s,)
        m = int(c)
        self.centre = np.array([index[p] for p in ((m, m), (m + 1, m), (m, m + 1), (m + 1, m + 1))])

    def grid(self, values: np.ndarray) -> np.ndarray:
        """Per-cell values laid out on the square grid (0 outside the disc), for drawing."""
        out = np.zeros((self.size, self.size))
        out[self.xy[:, 0], self.xy[:, 1]] = values[:len(self.xy)]
        return out


TASTE = cell.VARIABLES.index("t")      # the variable tastes kick
NOISE = 0.1                            # every life starts with each cell variable uniform in -NOISE..NOISE
DTYPE = cell.DTYPE                     # the cells' numbers (the world's are 64-bit)


@njit(cache=True)
def _random(rng, l):
    """The next uniform 0..1 from life l's own generator (splitmix64). Each life has its own stream, so a world's
    food and poison lives (the same seed) stay identical until the first taste."""
    rng[l] += np.uint64(0x9E3779B97F4A7C15)
    z = rng[l]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def start(state, n, rng):
    """Give every cell of every life small random variables, from each life's own generator."""
    lives = rng.shape[0]
    for l in range(lives):
        for a in range(state.shape[0]):
            for i in range(n):
                state[a, i * lives + l] = (_random(rng, l) * 2 - 1) * NOISE


@njit(cache=True)
def _place(blocks, l, b, pos, w, rng):
    size, radius = w[0], w[1]
    while True:
        x = radius + _random(rng, l) * (size - 2 * radius)
        y = radius + _random(rng, l) * (size - 2 * radius)
        if (x - pos[l, 0]) ** 2 + (y - pos[l, 1]) ** 2 > (4 * radius) ** 2:
            blocks[l, b, 0], blocks[l, b, 1] = x, y
            return


@njit(cache=True)
def tick(state, new, x, neighbours, weight, rule, angles, inputs, outputs, taste, centre, pos, blocks, food, counts,
         high, fired, rng, w):
    """One tick of L robots (one sheet each, the same rule) and their worlds: from state the next state is written into
    new (both (K, (n + 1) * L), as cell.step; x is scratch). Per life l, mutates pos[l], blocks[l], counts[l] (food
    eaten, poison eaten, thruster firings, cell spikes: any cell's v rising through FIRE_LEVEL), high[l] (actuator
    above FIRE_LEVEL), fired[l] (which actuators fired this tick) and rng[l]."""
    size, radius, reach, rate, push, kick, contact, middle, dt = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]
    lives = pos.shape[0]
    n = neighbours.shape[0]
    for l in range(lives):                                     # senses
        for s in range(len(angles)):
            fx, fy = math.cos(angles[s]), math.sin(angles[s])
            for colour in range(2):
                seen = 0.0
                for b in range(blocks.shape[1]):
                    if blocks[l, b, 2] == colour:
                        dx, dy = blocks[l, b, 0] - pos[l, 0], blocks[l, b, 1] - pos[l, 1]
                        d = math.sqrt(dx * dx + dy * dy) + 1e-9
                        seen += max(0.0, (fx * dx + fy * dy) / d) / (1 + (d / reach) ** 2)
                if _random(rng, l) < -math.expm1(-rate * min(1.0, seen) * dt):
                    c = inputs[s, colour] * lives + l
                    state[0, c] = min(1.0, state[0, c] + kick)
    cell.step(state, new, x, neighbours, weight, rule, dt)
    for c in range(n * lives):                                 # spikes, for the cost of activity
        if new[0, c] > cell.FIRE_LEVEL >= state[0, c]:
            counts[c % lives, 3] += 1
    for l in range(lives):
        for s in range(len(angles)):                           # actions
            now = new[0, outputs[s] * lives + l] > cell.FIRE_LEVEL
            fired[l, s] = now and not high[l, s]
            if fired[l, s]:
                pos[l, 0] -= push * math.cos(angles[s])
                pos[l, 1] -= push * math.sin(angles[s])
                counts[l, 2] += 1
            high[l, s] = now
        pos[l, 0] = min(size - radius, max(radius, pos[l, 0]))
        pos[l, 1] = min(size - radius, max(radius, pos[l, 1]))
        for b in range(blocks.shape[1]):                       # eating
            dx, dy = blocks[l, b, 0] - pos[l, 0], blocks[l, b, 1] - pos[l, 1]
            if dx * dx + dy * dy < (radius + 0.5) ** 2:
                good = blocks[l, b, 2] == food[l] or food[l] == BOTH
                counts[l, 0 if good else 1] += 1
                facing, best = 0, -2.0
                for s in range(len(angles)):
                    along = (math.cos(angles[s]) * dx + math.sin(angles[s]) * dy) / (math.sqrt(dx * dx + dy * dy) + 1e-9)
                    if along > best:
                        facing, best = s, along
                sign = 1.0 if good else -1.0
                c = taste[facing] * lives + l
                new[TASTE, c] = min(1.0, max(-1.0, new[TASTE, c] + sign * contact))
                for t in range(len(centre)):
                    c = centre[t] * lives + l
                    new[TASTE, c] = min(1.0, max(-1.0, new[TASTE, c] + sign * middle))
                _place(blocks, l, b, pos, w, rng)


@njit(cache=True)
def live(a, b, x, rule, neighbours, weight, angles, inputs, outputs, taste, centre, pos, blocks, food, w, ticks, rng):
    """Whole lives of L robots, one per world, all with one rule; a, b and x are the working arrays (as cell.step).
    Returns (food, poison, thruster firings, cell spikes) for each part of each life, shape (L, PARTS, 4). Moves pos
    and blocks."""
    lives = pos.shape[0]
    start(a, neighbours.shape[0], rng)
    counts = np.zeros((lives, PARTS, 4), dtype=np.int64)
    high = np.zeros((lives, len(angles)), dtype=np.bool_)
    fired = np.zeros((lives, len(angles)), dtype=np.bool_)
    for t in range(ticks):
        part = t * PARTS // ticks
        tick(a, b, x, neighbours, weight, rule, angles, inputs, outputs, taste, centre, pos, blocks, food,
             counts[:, part], high, fired, rng, w)
        a, b = b, a
    return counts


def working(n: int, lives: int):
    """The arrays cell.step works in, for n cells and L lives: two states and its scratch."""
    return (np.zeros((cell.K, (n + 1) * lives), dtype=DTYPE), np.zeros((cell.K, (n + 1) * lives), dtype=DTYPE),
            np.zeros((cell.N_INPUTS + cell.K, lives), dtype=DTYPE))


def lifetimes(rule, geometry: Geometry, task: str, worlds, world: World, duration: float) -> dict:
    """One life of a duration of simulated time in each of worlds ((seed, food) pairs), all with one rule, run
    together. Returns arrays over lives: counts (L, PARTS, 4) as `live`; start and end positions (L, 2); the blocks at the
    start and at the end (L, blocks, 3)."""
    world = world.for_task(task)
    setups = [setup(task, s, food, world) for s, food in worlds]
    pos = np.array([p for p, _ in setups])
    blocks = np.array([b for _, b in setups])
    start_pos, first = pos.copy(), blocks.copy()
    g = geometry
    rng = np.array([s for s, _ in worlds], dtype=np.uint64)
    counts = live(*working(g.sheet.n, len(worlds)), np.asarray(rule, dtype=DTYPE), g.sheet.neighbours, g.sheet.weight,
                  g.angles, g.inputs, g.outputs, g.taste, g.centre, pos, blocks,
                  np.array([f for _, f in worlds], dtype=float), world.array(), world.ticks(duration), rng)
    return {"counts": counts, "start": start_pos, "end": pos, "first": first, "last": blocks}


def setup(task: str, seed_: int, food: int, world: World = World()):
    """The robot's start and the blocks for a task: (pos, blocks)."""
    rng = np.random.default_rng(seed_)
    pos = np.array([world.size / 2, world.size / 2])
    if task == "move":
        return pos, np.zeros((0, 3))
    if task in ("approach", "taste"):
        angle, distance = rng.uniform(0, 2 * math.pi), rng.uniform(6, 12)
        colour = food if task == "approach" else RED
        return pos, np.array([[pos[0] + distance * math.cos(angle), pos[1] + distance * math.sin(angle), colour]])
    per_colour = 1 if task == "choose" else 4               # forage, graze, discriminate: 4 of each
    blocks = []
    while len(blocks) < 2 * per_colour:
        xy = rng.uniform(world.radius, world.size - world.radius, 2)
        if np.hypot(*(xy - pos)) > 5:
            blocks.append([xy[0], xy[1], len(blocks) // per_colour])
    return pos, np.array(blocks, dtype=float)


class Robot:
    """The robot for the app: one life, one tick at a time."""

    def __init__(self, rule: np.ndarray, task: str = "forage", seed_: int = 0, food: int = RED,
                 body: Body = Body(), world: World = World()):
        world = world.for_task(task)
        self.rule, self.task, self.food, self.world = np.asarray(rule, dtype=DTYPE), task, food, world
        self.geometry = g = Geometry(body)
        pos, blocks = setup(task, seed_, food, world)
        self._pos, self._blocks = pos[None].copy(), blocks[None].copy()
        self._food = np.array([food], dtype=float)
        self.rng = np.array([seed_], dtype=np.uint64)
        self.state, self.new, self.x = working(g.sheet.n, 1)
        start(self.state, g.sheet.n, self.rng)
        self._counts = np.zeros((1, 4), dtype=np.int64)
        self.high = np.zeros((1, len(g.angles)), dtype=np.bool_)
        self.fired = np.zeros((1, len(g.angles)), dtype=np.bool_)
        self.fired_at = np.full(len(g.angles), -1e9)
        self.ticks = 0

    pos = property(lambda self: self._pos[0])
    blocks = property(lambda self: self._blocks[0])
    counts = property(lambda self: self._counts[0])

    def step(self) -> None:
        g = self.geometry
        tick(self.state, self.new, self.x, g.sheet.neighbours, g.sheet.weight, self.rule, g.angles, g.inputs,
             g.outputs, g.taste, g.centre, self._pos, self._blocks, self._food, self._counts, self.high, self.fired,
             self.rng, self.world.array())
        self.state, self.new = self.new, self.state
        self.ticks += 1
        self.fired_at[self.fired[0]] = self.ticks
