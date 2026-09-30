"""A round robot run by a sheet of identical cells, in a walled arena with red and blue blocks.

In each run one colour is food and the other poison; the robot has to learn which, within the run, from the
consequences of eating. Everything the cells do is physics.py; outside the cells there is only the body:

  - the sheet is a disc of normal cells inside the round robot (random weights, START ± SPREAD)
  - S sensors sit evenly round the rim, each a red and a blue patch of input cells (a short radial stub, so a pulse
    enters as a front), struck at random at a rate that rises with that colour's intensity where the sensor is
    (so the difference between sensors says where the colour is)
  - A actuators sit evenly round the rim, between the sensors; each time an actuator cell ignites, the robot is
    pushed a little towards that side (it slides; it doesn't turn)
  - beside each actuator are a GOOD and a BAD teacher cell; eating food strikes every GOOD one, poison every BAD one
  - blocks are point sources: intensity 1 / (1 + (distance / REACH)^2); touching one eats it, and it reappears
    somewhere else
  - teaching is on throughout: the robot always learns, and cells fire by chance where they nearly fire

    python robot.py [SEEDS] [MINUTES]      learning against a no-learning control, red and blue food: the food
                                           share of meals in the first and last thirds of each run
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np

import physics
from model import SimulationConfig, State

DT = 1 / 50


@dataclass
class Body:
    grid: int = 32              # the sheet is grid x grid, with a disc of cells in it
    sensors: int = 8
    actuators: int = 8
    start: float = 0.36         # starting weights of the sheet's cells, plus a random spread
    spread: float = 0.02
    gap: float = 2.0            # how far round the rim each input cell is from its actuator, in cells


@dataclass
class World:
    size: float = 30.0          # the arena is size x size, walled
    radius: float = 2.0         # the robot's radius
    blocks: int = 4             # of each colour
    reach: float = 6.0          # how far a block's colour carries
    push: float = 3.0           # how far one actuator firing moves the robot
    rate: float = 0.15          # sensor pulses per second at full intensity (a cell takes seconds to refuel)


class Robot:
    """The body, the world and the sheet, advanced one time step at a time."""

    def __init__(self, seed: int = 0, food: str = 'red', body: Body | None = None, world: World | None = None,
                 config: SimulationConfig | None = None):
        self.body, self.world = body or Body(), world or World()
        self.food = food
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        self.state, self.sensor_cells, self.actuator_cells, self.teachers = self.build_sheet(seed)
        self.state.teaching = 1.0
        self.plastic = self.state.kind_array == physics.NORMAL
        w = self.world
        self.xy = np.array([w.size / 2, w.size / 2])
        self.blocks = [[self.place(), colour] for colour in ('red', 'blue') for _ in range(w.blocks)]
        self.t = 0.0
        self.eaten = []                       # (time, 'food' or 'poison')
        self.was_burning = np.zeros(len(self.actuator_cells), dtype=bool)
        self.fired = np.zeros(len(self.actuator_cells))      # time each actuator last fired, for display

    # --- the body ------------------------------------------------------------------------------------------------

    def build_sheet(self, seed):
        b = self.body
        n = b.grid
        c = (n - 1) / 2
        rim = n / 2 - 1.5
        s = State((n, n), seed=seed)
        rng = np.random.default_rng(seed)
        inside = lambda x, y: (x - c) ** 2 + (y - c) ** 2 <= (n / 2 - 1) ** 2
        for x in range(n):
            for y in range(n):
                if inside(x, y):
                    s.set_kind((x, y), physics.NORMAL)
                    s.weight_array[x, y] = max(0.0, b.start + rng.normal(0, b.spread))
        at = lambda angle, r: (int(round(c + r * math.cos(angle))), int(round(c + r * math.sin(angle))))
        sensors = []
        offset = b.gap / rim                  # the red and blue input cells sit either side of the actuator
        for i in range(b.sensors):
            angle = 2 * math.pi * i / b.sensors
            stub = lambda a: [at(a, rim - k) for k in range(3)]     # a short radial stub: a pulse enters as a front
            sensors.append({'angle': angle, 'red': stub(angle - offset), 'blue': stub(angle + offset)})
        actuators, teachers = [], {'good': [], 'bad': []}
        for i in range(b.actuators):
            angle = 2 * math.pi * i / b.actuators
            cell = at(angle, rim)
            actuators.append({'angle': angle, 'cell': cell})
            # a GOOD and a BAD teacher: the two cells next to the actuator that lie furthest outside the disc
            outside = sorted(((cell[0] + dx, cell[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                              if not inside(cell[0] + dx, cell[1] + dy)),
                             key=lambda xy: -((xy[0] - c) * math.cos(angle) + (xy[1] - c) * math.sin(angle)))
            for kind, xy in zip(('good', 'bad'), outside[:2]):
                s.set_kind(xy, physics.GOOD if kind == 'good' else physics.BAD)
                teachers[kind].append(xy)
        return s, sensors, actuators, teachers

    # --- the world -----------------------------------------------------------------------------------------------

    def place(self) -> np.ndarray:
        w = self.world
        while True:
            xy = self.rng.uniform(w.radius, w.size - w.radius, 2)
            if not hasattr(self, 'xy') or np.hypot(*(xy - self.xy)) > 4 * w.radius:
                return xy

    def intensity(self, angle: float, colour: str) -> float:
        """What a sensor facing `angle` sees of a colour: each block counts by how near it is and how squarely the
        sensor faces it."""
        w = self.world
        facing = np.array([math.cos(angle), math.sin(angle)])
        total = 0.0
        for bxy, c in self.blocks:
            if c == colour:
                d = bxy - self.xy
                distance = np.hypot(*d)
                total += max(0.0, facing @ d / distance) / (1 + (distance / w.reach) ** 2)
        return min(1.0, total)

    def step(self, dt: float = DT) -> None:
        s, w = self.state, self.world
        # senses: each input cell is struck at random, more often the stronger its colour is at the sensor
        for sensor in self.sensor_cells:
            for colour in ('red', 'blue'):
                if self.rng.random() < -math.expm1(-w.rate * self.intensity(sensor['angle'], colour) * dt):
                    for xy in sensor[colour]:
                        s.strike(xy, self.config)
        s.update(dt, self.config)
        self.t += dt
        # actions: each actuator that has just ignited pushes the robot towards its side
        for i, actuator in enumerate(self.actuator_cells):
            burning = s.flame_array[actuator['cell']] > 0
            if burning and not self.was_burning[i]:
                self.xy += w.push * np.array([math.cos(actuator['angle']), math.sin(actuator['angle'])])
                self.fired[i] = self.t
            self.was_burning[i] = burning
        self.xy = np.clip(self.xy, w.radius, w.size - w.radius)
        # eating: touching a block eats it; the teacher cells report what it was, and it reappears elsewhere
        for block in self.blocks:
            if np.hypot(*(block[0] - self.xy)) < w.radius + 0.5:
                good = block[1] == self.food
                self.eaten.append((self.t, 'food' if good else 'poison'))
                for xy in self.teachers['good' if good else 'bad']:
                    s.strike(xy, self.config)
                block[0] = self.place()

    def counts(self, since: float = 0.0, until: float = math.inf) -> tuple[int, int]:
        """Food and poison eaten in the time window."""
        window = [kind for t, kind in self.eaten if since <= t < until]
        return window.count('food'), window.count('poison')


def run(seed: int, food: str, minutes: float, params: dict | None = None, world: dict | None = None) -> dict:
    robot = Robot(seed, food, world=World(**(world or {})),
                  config=SimulationConfig(dict(physics.DEFAULT_PARAMS, **(params or {}))))
    for _ in range(round(minutes * 60 / DT)):
        robot.step()
    third = minutes * 60 / 3
    return {'seed': seed, 'food': food, 'total': robot.counts(), 'first': robot.counts(0, third),
            'last': robot.counts(2 * third)}


class Demo:
    """The robot for the app: one time step at a time, with a report every REPORT seconds."""
    REPORT = 30.0

    def __init__(self, seed: int, food: str = 'red', config: SimulationConfig | None = None):
        self.robot = Robot(seed, food, config=config)
        self.state, self.plastic = self.robot.state, self.robot.plastic
        self.at = {f"s{i}": sensor['red'][0] for i, sensor in enumerate(self.robot.sensor_cells)}
        self.at.update({f"m{i}": actuator['cell'] for i, actuator in enumerate(self.robot.actuator_cells)})
        self.max_temperature = self.robot.config.get('T_MAX')
        self.trials = 0
        self.next_report = self.REPORT

    @property
    def temperature(self) -> np.ndarray:
        return np.where(self.plastic, self.robot.config.get('T_BASE') + self.state.heat_array, 0.0)

    def step(self, dt: float, config: SimulationConfig) -> dict | None:
        self.robot.step(dt)
        if self.robot.t < self.next_report:
            return None
        self.trials += 1
        since = self.next_report - self.REPORT
        self.next_report += self.REPORT
        food, poison = self.robot.counts(since)
        total_food, total_poison = self.robot.counts()
        return {'text': f"t {self.robot.t:.0f}s ({self.robot.food} is food): last {self.REPORT:.0f}s ate {food} food, "
                        f"{poison} poison; total {total_food} food, {total_poison} poison"}


NO_LEARNING = {'CREDIT': 0.0, 'PUNISH': 0.0, 'BLAME': 0.0, 'FIRE_COST': 0.0}


def compare(seeds, minutes: float, world: dict | None = None) -> None:
    """Learning against a no-learning control, red and blue food: the food share of meals in the first and last
    thirds of each run."""
    from multiprocessing import Pool
    jobs = [(s, f, minutes, p, world) for p in (None, NO_LEARNING) for f in ('red', 'blue') for s in seeds]
    with Pool() as pool:
        results = pool.starmap(run, jobs)
    share = lambda food, poison: food / max(1, food + poison)
    for label, params in (('learning', None), ('no learning', NO_LEARNING)):
        rows = [r for (_, _, _, p, _), r in zip(jobs, results) if p is params]
        for r in rows:
            print(f"{label:11s} seed {r['seed']} {r['food']:4s} food: first third {r['first']}, last third {r['last']}")
        first = tuple(int(n) for n in np.sum([r["first"] for r in rows], axis=0))
        last = tuple(int(n) for n in np.sum([r["last"] for r in rows], axis=0))
        print(f"{label:11s} total: first third {first} ({share(*first):.0%} food), "
              f"last third {last} ({share(*last):.0%} food)", flush=True)


if __name__ == '__main__':
    compare(range(int(sys.argv[1]) if len(sys.argv) > 1 else 4), float(sys.argv[2]) if len(sys.argv) > 2 else 30)
