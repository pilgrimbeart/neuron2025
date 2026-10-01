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
    if red is food and poison otherwise), choose (one red, one blue, one of them food), forage (several of each)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

import cell

TASKS = ("move", "approach", "taste", "choose", "forage")
RED, BLUE = 0, 1


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
    grow: float = 10.0          # simulated time between one cell's birth and the next's, while the sheet grows

    def array(self) -> np.ndarray:
        return np.array([self.size, self.radius, self.reach, self.rate, self.push, self.kick, self.taste_contact,
                         self.taste_centre, self.dt])

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
        self.sheet = cell.Sheet([[index.get((x + dx, y + dy)) for dx, dy in cell.OFFSETS] for x, y in self.xy])
        births = []                         # the order cells are born in: a square spiral out from the middle
        for p in spiral(int(c), int(c)):
            if p in index:
                births.append(index[p])
                if len(births) == len(self.xy):
                    break
        self.order = np.array(births)
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


def spiral(x: int, y: int):
    """Grid points in a square spiral out from (x, y): east 1, south 1, west 2, north 2, east 3, ..."""
    yield x, y
    step = 1
    while True:
        for dx, dy, length in ((1, 0, step), (0, 1, step), (-1, 0, step + 1), (0, -1, step + 1)):
            for _ in range(length):
                x, y = x + dx, y + dy
                yield x, y
        step += 2


TASTE = cell.VARIABLES.index("t")      # the variable tastes kick


@njit(cache=True)
def develop(rule, neighbours, order, grow_ticks, dt):
    """Grow the sheet: one cell (the first in order) at the start, and the next one every grow_ticks ticks, each
    born with every variable 0, until all are there. Nothing outside reaches the cells meanwhile, so for a given rule
    and dt the result is always the same. Returns the grown state, (K, n + 1)."""
    n = neighbours.shape[0]
    a = np.zeros((cell.K, n + 1))
    b = np.zeros_like(a)
    x = np.zeros((cell.N_INPUTS, n))
    alive = np.zeros(n, dtype=np.bool_)
    for k in range(n):
        alive[order[k]] = True
        for _ in range(grow_ticks):
            cell.step(a, b, x, neighbours, alive, rule, dt)
            a, b = b, a
    return a


@njit(cache=True)
def seed(seed_):
    np.random.seed(seed_)


@njit(cache=True)
def _place(blocks, b, pos, w):
    size, radius = w[0], w[1]
    while True:
        x = radius + np.random.random() * (size - 2 * radius)
        y = radius + np.random.random() * (size - 2 * radius)
        if (x - pos[0]) ** 2 + (y - pos[1]) ** 2 > (4 * radius) ** 2:
            blocks[b, 0], blocks[b, 1] = x, y
            return


@njit(cache=True)
def tick(state, new, x, neighbours, alive, rule, angles, inputs, outputs, taste, centre, pos, blocks, food, counts,
         high, fired, w):
    """One tick of the robot and its world: from state, the next state is written into new (x is scratch). Mutates
    pos, blocks, counts (food eaten, poison eaten, thruster firings, cell spikes: any cell's v rising through
    FIRE_LEVEL), high (actuator above FIRE_LEVEL) and fired
    (which actuators fired this tick)."""
    size, radius, reach, rate, push, kick, contact, middle, dt = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]
    for s in range(len(angles)):                               # senses
        fx, fy = math.cos(angles[s]), math.sin(angles[s])
        for colour in range(2):
            seen = 0.0
            for b in range(len(blocks)):
                if blocks[b, 2] == colour:
                    dx, dy = blocks[b, 0] - pos[0], blocks[b, 1] - pos[1]
                    d = math.sqrt(dx * dx + dy * dy) + 1e-9
                    seen += max(0.0, (fx * dx + fy * dy) / d) / (1 + (d / reach) ** 2)
            if np.random.random() < -math.expm1(-rate * min(1.0, seen) * dt):
                i = inputs[s, colour]
                state[0, i] = min(1.0, state[0, i] + kick)
    cell.step(state, new, x, neighbours, alive, rule, dt)
    for i in range(x.shape[1]):                                # spikes, for the cost of activity
        if new[0, i] > cell.FIRE_LEVEL >= state[0, i]:
            counts[3] += 1
    for s in range(len(angles)):                               # actions
        now = new[0, outputs[s]] > cell.FIRE_LEVEL
        fired[s] = now and not high[s]
        if fired[s]:
            pos[0] -= push * math.cos(angles[s])
            pos[1] -= push * math.sin(angles[s])
            counts[2] += 1
        high[s] = now
    pos[0] = min(size - radius, max(radius, pos[0]))
    pos[1] = min(size - radius, max(radius, pos[1]))
    for b in range(len(blocks)):                               # eating
        dx, dy = blocks[b, 0] - pos[0], blocks[b, 1] - pos[1]
        if dx * dx + dy * dy < (radius + 0.5) ** 2:
            good = blocks[b, 2] == food
            counts[0 if good else 1] += 1
            facing, best = 0, -2.0
            for s in range(len(angles)):
                along = (math.cos(angles[s]) * dx + math.sin(angles[s]) * dy) / (math.sqrt(dx * dx + dy * dy) + 1e-9)
                if along > best:
                    facing, best = s, along
            sign = 1.0 if good else -1.0
            i = taste[facing]
            new[TASTE, i] = min(1.0, max(-1.0, new[TASTE, i] + sign * contact))
            for t in range(len(centre)):
                i = centre[t]
                new[TASTE, i] = min(1.0, max(-1.0, new[TASTE, i] + sign * middle))
            _place(blocks, b, pos, w)


@njit(cache=True)
def live(rule, grown, neighbours, angles, inputs, outputs, taste, centre, pos, blocks, food, w, ticks, seed_):
    """A whole life, starting from the grown sheet; returns (food, poison, thruster firings, cell spikes) for each
    half of it, shape (2, 4). Moves pos and blocks."""
    n = neighbours.shape[0]
    seed(seed_)
    a = grown.copy()
    b = np.zeros_like(a)
    alive = np.ones(n, dtype=np.bool_)
    x = np.zeros((cell.N_INPUTS, n))
    counts = np.zeros((2, 4), dtype=np.int64)
    high = np.zeros(len(angles), dtype=np.bool_)
    fired = np.zeros(len(angles), dtype=np.bool_)
    for t in range(ticks):
        half = counts[0] if t < ticks // 2 else counts[1]
        tick(a, b, x, neighbours, alive, rule, angles, inputs, outputs, taste, centre, pos, blocks, food, half, high,
             fired, w)
        a, b = b, a
    return counts


def grow(rule, geometry: Geometry, world: World) -> np.ndarray:
    """The grown sheet for a rule, at the world's tick size (the same for every life, so grow it once)."""
    return develop(rule, geometry.sheet.neighbours, geometry.order, world.ticks(world.grow), world.dt)


def lifetime(rule, geometry: Geometry, task: str, seed_: int, food: int, world: World, duration: float, grown=None):
    """A whole life of a duration of simulated time, in a fresh world, after the sheet has grown (pass grown, from
    `grow`, to reuse it): (per-half counts, as `live`; start position; final position; the blocks at the start; the
    blocks at the end)."""
    if grown is None:
        grown = grow(rule, geometry, world)
    pos, blocks = setup(task, seed_, food, world)
    start_pos, first = pos.copy(), blocks.copy()
    g = geometry
    counts = live(rule, grown, g.sheet.neighbours, g.angles, g.inputs, g.outputs, g.taste, g.centre, pos, blocks,
                  food, world.array(), world.ticks(duration), seed_)
    return counts, start_pos, pos, first, blocks


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
    per_colour = 1 if task == "choose" else 4
    blocks = []
    while len(blocks) < 2 * per_colour:
        xy = rng.uniform(world.radius, world.size - world.radius, 2)
        if np.hypot(*(xy - pos)) > 5:
            blocks.append([xy[0], xy[1], len(blocks) // per_colour])
    return pos, np.array(blocks, dtype=float)


class Robot:
    """The robot for the app: one tick at a time, growing the sheet first (the robot waits, unborn, meanwhile)."""

    def __init__(self, rule: np.ndarray, task: str = "forage", seed_: int = 0, food: int = RED,
                 body: Body = Body(), world: World = World()):
        self.rule, self.task, self.food, self.world = rule, task, food, world
        self.geometry = g = Geometry(body)
        self.pos, self.blocks = setup(task, seed_, food, world)
        seed(seed_)
        n = g.sheet.n
        self.state = np.zeros((cell.K, n + 1))
        self.new = np.zeros_like(self.state)
        self.x = np.zeros((cell.N_INPUTS, n))
        self.alive = np.zeros(n, dtype=np.bool_)
        self.born = 0                       # cells born so far; the robot lives once all n are
        self.grow_ticks = world.ticks(world.grow)
        self.counts = np.zeros(4, dtype=np.int64)
        self.high = np.zeros(len(g.angles), dtype=np.bool_)
        self.fired = np.zeros(len(g.angles), dtype=np.bool_)
        self.fired_at = np.full(len(g.angles), -1e9)
        self.ticks = 0                      # ticks of life, after growing
        self.growing_ticks = 0

    @property
    def growing(self) -> bool:
        return self.born < self.geometry.sheet.n or self.growing_ticks % self.grow_ticks

    def step(self) -> None:
        g = self.geometry
        if self.growing:
            if self.growing_ticks % self.grow_ticks == 0:
                self.alive[g.order[self.born]] = True
                self.born += 1
            cell.step(self.state, self.new, self.x, g.sheet.neighbours, self.alive, self.rule, self.world.dt)
            self.state, self.new = self.new, self.state
            self.growing_ticks += 1
            return
        tick(self.state, self.new, self.x, g.sheet.neighbours, self.alive, self.rule, g.angles, g.inputs,
             g.outputs, g.taste, g.centre, self.pos, self.blocks, self.food, self.counts, self.high, self.fired,
             self.world.array())
        self.state, self.new = self.new, self.state
        self.ticks += 1
        self.fired_at[self.fired] = self.ticks
