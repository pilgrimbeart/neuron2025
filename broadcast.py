"""Experiments: can a random medium learn, from a broadcast reward signal alone, to do something useful?

Nothing here is designed junction by junction. A random patch of cells sits between an input and two outputs;
after each stimulus, a global reward (+) or punishment (-) is broadcast when an output fires, and every cell
changes its weight in proportion to its own recent involvement (its light trace). The physics is the standard
physics.py with its built-in learning rule switched off.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage

import physics
from gates import hline
from model import SimulationConfig, State

MAX_WAIT = 40.0


@dataclass
class Medium:
    size: int = 48
    density: float = 0.62
    seed: int = 0
    box: tuple[int, int, int, int] = (8, 8, 39, 39)       # x0, y0, x1, y1 of the random region

    def build(self) -> tuple[State, dict[str, tuple[int, int]]]:
        rng = np.random.default_rng(self.seed)
        s = State((self.size, self.size))
        x0, y0, x1, y1 = self.box
        mask = rng.random((x1 - x0 + 1, y1 - y0 + 1)) < self.density
        for x, y in zip(*np.nonzero(mask)):
            s.set_kind((x0 + int(x), y0 + int(y)), physics.NORMAL)
        n = self.size - 1
        mid, top, bot = (y0 + y1) // 2, y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        for xy in hline(mid, 0, x0 + 1) | hline(top, x1 - 1, n) | hline(bot, x1 - 1, n):
            s.set_kind(xy, physics.NORMAL)
        plastic = np.zeros(s.grid_size, dtype=bool)
        plastic[x0 + 2:x1 - 1, y0:y1 + 1] = True          # the random region learns; the input and output wires don't
        plastic &= s.kind_array != physics.EMPTY
        return s, {'a': (0, mid), 'x': (n, top), 'y': (n, bot)}, plastic



@dataclass
class Wires:
    """A random network of one-cell-wide wires grown as random walks from random starts. The only rule: a wire may
    touch another wire diagonally (a potential synapse: it conducts only if the receiving cell's weight is high
    enough) but never orthogonally (which would always conduct). a's input wire continues into the network;
    the x and y output wires reach back into it from the right."""
    size: int = 48
    wires: int = 120
    length: int = 30           # longest a wire grows
    turn: float = 0.3          # chance per step of picking a new random heading
    seed: int = 0
    box: tuple[int, int, int, int] = (6, 6, 41, 41)
    reach: int = 12            # how far the input and output wires extend into the network

    def build(self):
        rng = np.random.default_rng(self.seed)
        x0, y0, x1, y1 = self.box
        n = self.size - 1
        owner = -np.ones((self.size, self.size), dtype=int)

        def free(xy, wire, prev):
            x, y = xy
            if not (x0 <= x <= x1 and y0 <= y <= y1) or owner[xy] != -1:
                return False
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < self.size and 0 <= ny < self.size and owner[nx, ny] != -1 and (nx, ny) != prev:
                    return False          # no orthogonal contact with any other cell, nor with itself except the previous
            return True

        def lay(cells, wire):
            for xy in cells:
                owner[xy] = wire

        mid, top, bot = (y0 + y1) // 2, y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        a_line = [(x, mid) for x in range(0, x0 + self.reach)]
        x_line = [(x, top) for x in range(x1 - self.reach, n + 1)]
        y_line = [(x, bot) for x in range(x1 - self.reach, n + 1)]
        for w, cells in enumerate((a_line, x_line, y_line)):
            for xy in cells:
                owner[xy] = w
        moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for w in range(3, 3 + self.wires):
            start = (int(rng.integers(x0, x1 + 1)), int(rng.integers(y0, y1 + 1)))
            if not free(start, w, None):
                continue
            owner[start] = w
            path, heading = [start], moves[int(rng.integers(4))]
            while len(path) < self.length:
                if rng.random() < self.turn:
                    heading = moves[int(rng.integers(4))]
                x, y = path[-1]
                options = [heading] + [m for m in moves if m != heading]
                step = next(((x + dx, y + dy) for dx, dy in options if free((x + dx, y + dy), w, path[-1])), None)
                if step is None:
                    break
                owner[step] = w
                path.append(step)
        s = State((self.size, self.size))
        for xy in zip(*np.nonzero(owner >= 0)):
            s.set_kind((int(xy[0]), int(xy[1])), physics.NORMAL)
        plastic = owner >= 3
        inside = np.zeros_like(plastic)
        inside[x0:x1 + 1, y0:y1 + 1] = True
        plastic |= inside & ((owner == 1) | (owner == 2))    # the outputs' synapses are learnable too
        return s, {'a': (0, mid), 'x': (n, top), 'y': (n, bot)}, plastic


@dataclass
class Sheet:
    """Every cell in the region is a normal, plastic cell: weights alone decide where activity goes. a's input wire
    enters from the left edge; the x and y output wires leave from the right edge. Starting weights get a small
    random spread, which breaks the symmetry between x and y."""
    size: int = 48
    seed: int = 0
    box: tuple[int, int, int, int] = (6, 6, 41, 41)
    spread: float = 0.1
    entry: int = 0             # how many cells each input wire extends into the sheet (a front rather than a single contact)
    two_inputs: bool = False   # inputs a (upper) and b (lower) instead of a single a in the middle

    def build(self):
        x0, y0, x1, y1 = self.box
        n = self.size - 1
        mid, top, bot = (y0 + y1) // 2, y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        inputs = {'a': top, 'b': bot} if self.two_inputs else {'a': mid}
        s = State((self.size, self.size))
        cells = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
        cells += [(x, row) for row in inputs.values() for x in range(0, x0)]
        cells += [(x, top) for x in range(x1 + 1, n + 1)] + [(x, bot) for x in range(x1 + 1, n + 1)]
        for xy in cells:
            s.set_kind(xy, physics.NORMAL)
        plastic = np.zeros(s.grid_size, dtype=bool)
        plastic[x0:x1 + 1, y0:y1 + 1] = True
        for row in inputs.values():
            plastic[x0:x0 + self.entry, row] = False       # input wires inside the sheet stay fixed
        self.jitter = np.random.default_rng(self.seed).normal(0, self.spread, s.grid_size) * plastic
        at = {label: (0, row) for label, row in inputs.items()}
        return s, {**at, 'x': (n, top), 'y': (n, bot)}, plastic

@dataclass
class Rule:
    rate: float = 0.5          # weight change per unit of light trace per reward
    trace: str = 'fire'        # eligibility: 'light' (physics light trace) or 'fire' (own firing, fading over elig_time)
    elig_time: float = 4.0     # seconds for the firing eligibility trace to fade
    homeo: float = 0.0         # homeostasis: per trial, weight += homeo * (target - fired this trial)
    lit_only: bool = False     # homeostasis only for cells that received light this trial (no evidence, no change)
    down_only: bool = False    # homeostasis only ever lowers weights (quietens over-active cells, never re-sensitises)
    conserve: float = 0.0      # if > 0: each reward only redistributes weight within this radius (cells), zero-sum locally
    rpe: float = 0.0           # if > 0: broadcast surprise, sign * (1 - p), where p tracks how often that output fires
                               # (updated per trial at this rate); an output that always fires then teaches nothing
    global_homeo: float = 0.0  # per trial, shift every plastic weight by this * (target - fraction of cells that fired):
                               # one broadcast "how busy" number; relative differences (the learning) are preserved
    explore: float = 0.0       # each trial, every plastic cell's weight gets a fresh random offset of this spread
                               # (seeded, so reproducible), removed after the trial: trials differ, so reward can select
    target: float = 0.2        # the fraction of trials each cell aims to fire in
    start: float = 1.0         # initial weight of every cell


REWARDS = {'x': +1.0, 'y': -1.0}      # R1: reward when x fires, punish when y fires


def broadcast_config(gain: float = 1.0) -> SimulationConfig:
    """The shared physics, with its built-in (transducer) learning switched off and weights allowed down to 0
    (a cell with weight 0 never ignites: effectively switched off)."""
    p = dict(physics.DEFAULT_PARAMS, LEARN_RATE=0.0, UNLEARN_RATE=0.0, WEIGHT_MIN=0.0)
    p['COUPLING_GAIN'] *= gain
    return SimulationConfig(p)


class Trainer:
    """Runs trials one time step at a time, so the app can show them live and scripts can run them fast.

    Each trial: strike a; run; the first time each output fires, broadcast its reward to every plastic cell in
    proportion to its eligibility; once the grid is quiet (or after MAX_WAIT), apply homeostasis, clear the
    traces, and start the next trial."""

    def __init__(self, state: State, at: dict, plastic: np.ndarray, rule: Rule, rewards: dict = REWARDS,
                 jitter: np.ndarray | None = None, seed: int = 0, schedule: list | None = None):
        """schedule: [(input label, rewards)] conditions, cycled one per trial; default [('a', rewards)]."""
        self.state, self.at, self.plastic, self.rule = state, at, plastic, rule
        self.schedule = schedule or [('a', rewards)]
        self.learn = True
        self.trials = 0
        self.expect = {(i, k): 0.0 for i, r in self.schedule for k in r}
        self.rng = np.random.default_rng(1000 + seed)
        self.offset = np.zeros(state.grid_size)
        state.weight_array[plastic] = rule.start + (0 if jitter is None else jitter[plastic])
        self.fire_trace = np.zeros(state.grid_size)
        self.start_trial()

    def start_trial(self) -> None:
        self.t = 0.0
        self.input, self.rewards = self.schedule[self.trials % len(self.schedule)]
        if self.learn and self.rule.explore:
            self.offset = self.rng.normal(0.0, self.rule.explore, self.state.grid_size) * self.plastic
        else:
            self.offset = np.zeros(self.state.grid_size)
        self.state.weight_array += self.offset
        self.fired = {k: False for k in self.rewards}
        self.fired_cells = np.zeros(self.state.grid_size, dtype=bool)
        self.peak_light = np.zeros(self.state.grid_size)

    def test_mode(self, input_label: str) -> None:
        """Stop learning and exploring: remove the pending trial's random offset, and run only this input from now on."""
        self.state.weight_array -= self.offset
        self.offset = np.zeros(self.state.grid_size)
        self.learn = False
        self.schedule = [(input_label, {'x': 0.0, 'y': 0.0})]
        self.trials = 0
        self.start_trial()

    def eligibility(self) -> np.ndarray:
        return self.state.light_trace_array if self.rule.trace == 'light' else self.fire_trace

    def reward(self, sign: float, config: SimulationConfig) -> None:
        s, m, p = self.state, self.plastic, config.vars
        change = sign * self.rule.rate * self.eligibility() * m
        if self.rule.conserve > 0:
            local = scipy.ndimage.gaussian_filter(change, self.rule.conserve, mode='constant')
            area = scipy.ndimage.gaussian_filter(m.astype(float), self.rule.conserve, mode='constant')
            change = (change - np.where(area > 0, local / np.maximum(area, 1e-9), 0.0)) * m
        base = s.weight_array - self.offset
        base[m] = np.clip(base[m] + change[m], p['WEIGHT_MIN'], p['WEIGHT_MAX'])
        s.weight_array[:] = base + self.offset

    def step(self, dt: float, config: SimulationConfig) -> dict | None:
        """Advance one time step. Returns the trial's summary when a trial ends, else None."""
        s = self.state
        if self.t == 0.0:
            s.strike(self.at[self.input], config)
        before = {k: s.flame_array[self.at[k]] > 0 for k in self.rewards}
        s.update(dt, config)
        self.t += dt
        self.fired_cells |= s.flame_array > 0
        self.peak_light = np.maximum(self.peak_light, s.illumination_array)
        self.fire_trace = np.maximum((s.flame_array > 0).astype(float), self.fire_trace * np.exp(-dt / self.rule.elig_time))
        for k, sign in self.rewards.items():
            if s.flame_array[self.at[k]] > 0 and not before[k] and not self.fired[k]:
                self.fired[k] = True
                if self.learn and sign:
                    self.reward(sign * (1 - self.expect[(self.input, k)]) if self.rule.rpe else sign, config)
        cells = s.kind_array != physics.EMPTY
        quiet = not s.flame_array.any() and s.energy_array[cells].min() >= 1.0
        if not (quiet or self.t >= MAX_WAIT):
            return None
        return self.end_trial(quiet, config)

    def end_trial(self, quiet: bool, config: SimulationConfig) -> dict:
        s, p = self.state, config.vars
        s.weight_array -= self.offset
        if self.learn and self.rule.global_homeo:
            m = self.plastic
            s.weight_array[m] = np.clip(s.weight_array[m] + self.rule.global_homeo * (self.rule.target - self.fired_cells[m].mean()),
                                        p['WEIGHT_MIN'], p['WEIGHT_MAX'])
        if self.learn and self.rule.homeo:
            m = self.plastic
            if self.rule.lit_only:
                m = m & (self.fired_cells | (self.peak_light >= 0.25 * p['MIN_STRIKE']))
            change = self.rule.homeo * (self.rule.target - self.fired_cells[m])
            if self.rule.down_only:
                change = np.minimum(change, 0.0)
            s.weight_array[m] = np.clip(s.weight_array[m] + change, p['WEIGHT_MIN'], p['WEIGHT_MAX'])
        s.light_trace_array[:] = 0
        s.teach_trace_array[:] = 0
        self.fire_trace[:] = 0
        self.trials += 1
        if self.rule.rpe:
            for k in self.rewards:
                self.expect[(self.input, k)] += self.rule.rpe * (self.fired[k] - self.expect[(self.input, k)])
        summary = {'input': self.input, **self.fired, 'quiet': quiet, 'active': float(self.fired_cells[self.plastic].mean())}
        self.start_trial()
        return summary

    def trial(self, config: SimulationConfig, dt: float = 1 / 50) -> dict:
        while (summary := self.step(dt, config)) is None:
            pass
        return summary


def score(medium, rule, trials: int = 150, rewarded: bool = True) -> dict:
    """Train on R1 for a number of trials, then test without learning. rewarded=False is the control: the same,
    with rewards switched off."""
    state, at, plastic = medium.build()
    config = broadcast_config()
    trainer = Trainer(state, at, plastic, rule, REWARDS if rewarded else {'x': 0.0, 'y': 0.0}, getattr(medium, 'jitter', None),
                      seed=getattr(medium, 'seed', 0))
    for _ in range(trials):
        trainer.trial(config)
    trainer.test_mode('a')
    tests = [trainer.trial(config) for _ in range(4)]
    wins = sum(t['x'] and not t['y'] for t in tests)
    return {'wins': wins, 'x': sum(t['x'] for t in tests), 'y': sum(t['y'] for t in tests),
            'active': sum(t['active'] for t in tests) / 4}



MAPPINGS = {'straight': {'a': 'x', 'b': 'y'}, 'crossed': {'a': 'y', 'b': 'x'}}


def mapping_schedule(mapping: str) -> list:
    """One condition per input: reward its target output, punish the other."""
    return [(i, {o: (+1.0 if o == target else -1.0) for o in ('x', 'y')}) for i, target in MAPPINGS[mapping].items()]


def score_mapping(seed: int, mapping: str, rule: Rule, trials: int = 300, rewarded: bool = True, sheet: dict | None = None) -> dict:
    """Train a two-input sheet on a mapping (alternating a and b trials), then test each input alone."""
    m = Sheet(seed=seed, two_inputs=True, **(sheet or {}))
    state, at, plastic = m.build()
    config = broadcast_config()
    schedule = mapping_schedule(mapping) if rewarded else [(i, {'x': 0.0, 'y': 0.0}) for i in ('a', 'b')]
    trainer = Trainer(state, at, plastic, rule, jitter=m.jitter, seed=seed, schedule=schedule)
    for _ in range(trials):
        trainer.trial(config)
    result = {}
    for i, target in MAPPINGS[mapping].items():
        other = 'y' if target == 'x' else 'x'
        trainer.test_mode(i)
        tests = [trainer.trial(config) for _ in range(2)]
        result[i] = ''.join(o for o in ('x', 'y') if tests[-1][o]) or '-'
        result[i + '_ok'] = all(t[target] and not t[other] for t in tests)
    result['ok'] = result['a_ok'] and result['b_ok']
    return result

# The live demo: a full sheet held near the edge of firing, with per-trial exploration and reward prediction error.
DEMO_RULE = Rule(rate=0.1, elig_time=2.0, start=0.4, explore=0.1, rpe=0.1)
DEMO_SHEET = dict(spread=0.15, entry=6)


def demo(seed: int, mapping: str | None = None) -> Trainer:
    """The live demo's trainer: R1 on a one-input sheet, or a two-input sheet trained on a mapping."""
    m = Sheet(seed=seed, two_inputs=mapping is not None, **DEMO_SHEET)
    state, at, plastic = m.build()
    schedule = mapping_schedule(mapping) if mapping else None
    return Trainer(state, at, plastic, dataclasses.replace(DEMO_RULE), jitter=m.jitter, seed=seed, schedule=schedule)


if __name__ == '__main__':
    state, at, plastic = Wires().build()
    config = broadcast_config()
    trainer = Trainer(state, at, plastic, DEMO_RULE)
    for n in range(40):
        r = trainer.trial(config)
        print(f"trial {n + 1}: x {r['x']:d} y {r['y']:d} active {r['active']:.2f}")
