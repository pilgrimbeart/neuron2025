"""Experiments: learning by "paying attention", with no global signal.

A teacher pays attention to the output it wants whenever that output fires. A cell that is paid attention passes it
back to its causes: the neighbours that fired first before it (their fire times lie on the steepest descent of the
wave's arrival time). Attention therefore runs back along the route the wave actually took, from the output to the
input, and never reaches cells that were merely active at the same time. Cells that are paid attention strengthen;
cells that fire and are not paid attention weaken a little (firing costs, being useful pays).

The medium is the full sheet from broadcast.py, with the standard physics and its built-in learning switched off.

    python attention.py [N]    score a -> x, then the straight and crossed mappings, on N random sheets, with controls
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

import physics
from broadcast import MAPPINGS, MAX_WAIT, Sheet, broadcast_config
from model import SimulationConfig, State

NEIGHBOURS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
# light reaches about two cells, so a cell with no earlier neighbour may have been lit from the next ring out
FAR_NEIGHBOURS = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2 and dx * dx + dy * dy <= 5]


@dataclass
class Rule:
    rate: float = 0.1       # weight gained by a cell each time it is paid attention
    cost: float = 0.002     # weight lost by a cell each time it fires and is not paid attention
    tol: float = 0.3        # a cause is any earlier neighbour within this fraction of the earliest one's lead
    start: float = 0.45     # starting weight of every sheet cell (plus the sheet's random spread)
    explore: float = 0.0    # each trial, every plastic weight gets a fresh random offset of this spread (seeded), removed
                            # after the trial: trials differ, so an output nobody reaches yet can be reached by chance


def causes(fired_at: np.ndarray, xy: tuple[int, int], tol: float) -> list[tuple[int, int]]:
    """The cells that fired first before xy among its neighbours (ties within tol of the earliest's lead); if none
    of its neighbours fired before it, among the cells two away, which light also reaches."""
    x, y = xy
    t = fired_at[xy]
    for ring in (NEIGHBOURS, FAR_NEIGHBOURS):
        earlier = [(x + dx, y + dy) for dx, dy in ring
                   if 0 <= x + dx < fired_at.shape[0] and 0 <= y + dy < fired_at.shape[1] and fired_at[x + dx, y + dy] < t]
        if earlier:
            lead = t - min(fired_at[n] for n in earlier)
            return [n for n in earlier if t - fired_at[n] >= (1 - tol) * lead]
    return []


def relay(fired_at: np.ndarray, start: tuple[int, int], tol: float) -> np.ndarray:
    """Every cell attention reaches from start, passing from each cell to its causes."""
    attended = np.zeros(fired_at.shape, dtype=bool)
    attended[start] = True
    todo = [start]
    while todo:
        for n in causes(fired_at, todo.pop(), tol):
            if not attended[n]:
                attended[n] = True
                todo.append(n)
    return attended


class Trainer:
    """Runs trials one time step at a time (same interface as broadcast.Trainer, so the app can show it live).

    Each trial: strike the input; the first time the output being paid attention to fires, relay attention back
    from it and strengthen every cell it reaches; once quiet, weaken every cell that fired unattended."""

    def __init__(self, state: State, at: dict, plastic: np.ndarray, rule: Rule, schedule: list, jitter: np.ndarray | None = None,
                 seed: int = 0, outputs: tuple[str, ...] = ('x', 'y')):
        """schedule: [(input label, output label to pay attention to, or None)], cycled one per trial."""
        self.state, self.at, self.plastic, self.rule, self.schedule = state, at, plastic, rule, schedule
        self.outputs = outputs
        self.learn = True
        self.trials = 0
        self.rng = np.random.default_rng(1000 + seed)
        self.offset = np.zeros(state.grid_size)
        state.weight_array[plastic] = rule.start + (0 if jitter is None else jitter[plastic])
        self.start_trial()

    def start_trial(self) -> None:
        self.t = 0.0
        self.input, self.watch = self.schedule[self.trials % len(self.schedule)]
        self.fired_at = np.full(self.state.grid_size, np.inf)
        self.fired = {k: False for k in self.outputs}
        self.attended = np.zeros(self.state.grid_size, dtype=bool)
        explore = self.rule.explore if self.learn else 0.0
        self.offset = self.rng.normal(0.0, explore, self.state.grid_size) * self.plastic if explore else np.zeros(self.state.grid_size)
        self.state.weight_array += self.offset

    def test_mode(self, input_label: str) -> None:
        self.state.weight_array -= self.offset
        self.learn = False
        self.schedule = [(input_label, None)]
        self.trials = 0
        self.start_trial()

    def step(self, dt: float, config: SimulationConfig) -> dict | None:
        s = self.state
        if self.t == 0.0:
            s.strike(self.at[self.input], config)
        s.update(dt, config)
        self.t += dt
        self.fired_at[(s.flame_array > 0) & np.isinf(self.fired_at)] = self.t
        for k in self.fired:
            if not self.fired[k] and s.flame_array[self.at[k]] > 0:
                self.fired[k] = True
                if self.learn and k == self.watch:
                    self.attended = relay(self.fired_at, self.at[k], self.rule.tol)
                    m = self.attended & self.plastic
                    s.weight_array[m] = np.minimum(s.weight_array[m] + self.rule.rate, config.get('WEIGHT_MAX'))
        cells = s.kind_array != physics.EMPTY
        quiet = not s.flame_array.any() and s.energy_array[cells].min() >= 1.0
        if not (quiet or self.t >= MAX_WAIT):
            return None
        fired_cells = np.isfinite(self.fired_at)
        s.weight_array -= self.offset
        if self.learn:
            m = fired_cells & ~self.attended & self.plastic
            s.weight_array[m] = np.maximum(s.weight_array[m] - self.rule.cost, config.get('WEIGHT_MIN'))
        s.light_trace_array[:] = 0
        s.teach_trace_array[:] = 0
        self.trials += 1
        summary = {'input': self.input, **self.fired, 'quiet': quiet, 'active': float(fired_cells[self.plastic].mean()),
                   'attended': int((self.attended & self.plastic).sum())}
        self.last_attended, self.last_fired_at = self.attended, self.fired_at
        self.start_trial()
        return summary

    def trial(self, config: SimulationConfig, dt: float = 1 / 50) -> dict:
        while (summary := self.step(dt, config)) is None:
            pass
        return summary


def schedule(mapping: str | None, attend: bool = True) -> list:
    """a -> x alone (mapping None), or both inputs of a mapping alternately. attend=False is the control."""
    pairs = {'a': 'x'} if mapping is None else MAPPINGS[mapping]
    return [(i, o if attend else None) for i, o in pairs.items()]


def demo(seed: int, mapping: str | None = None, rule: Rule | None = None, attend: bool = True) -> Trainer:
    m = Sheet(seed=seed, two_inputs=mapping is not None, spread=0.15, entry=6)
    state, at, plastic = m.build()
    for o in ('x', 'y'):
        plastic[m.box[2] + 1, at[o][1]] = True       # each output's first cell is its synapse onto the sheet
    return Trainer(state, at, plastic, dataclasses.replace(rule or Rule()), schedule(mapping, attend), jitter=m.jitter, seed=seed)


def score(seed: int, mapping: str | None = None, rule: Rule | None = None, trials: int = 100, attend: bool = True) -> dict:
    """Train, then test each input alone: it must reach its output and not the other."""
    trainer = demo(seed, mapping, rule, attend)
    config = broadcast_config()
    history = [trainer.trial(config) for _ in range(trials)]
    result = {'active': history[-1]['active']}
    for i, target in ({'a': 'x'} if mapping is None else MAPPINGS[mapping]).items():
        other = 'y' if target == 'x' else 'x'
        trainer.test_mode(i)
        tests = [trainer.trial(config) for _ in range(2)]
        result[i] = ''.join(o for o in ('x', 'y') if tests[-1][o]) or '-'
        result[i + '_ok'] = all(t[target] and not t[other] for t in tests)
        result[i + '_active'] = tests[-1]['active']
    result['ok'] = all(v for k, v in result.items() if k.endswith('_ok'))
    return result


if __name__ == '__main__':
    import sys
    from multiprocessing import Pool
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    tasks = (None, 'straight', 'crossed')
    jobs = [(s, mapping, attend) for mapping in tasks for attend in (True, False) for s in seeds]
    with Pool() as pool:
        results = pool.starmap(score, [(s, mapping, None, 200, attend) for s, mapping, attend in jobs])
    for mapping in tasks:
        for attend in (True, False):
            rs = [r for (s, m, a), r in zip(jobs, results) if m == mapping and a == attend]
            detail = ' '.join(','.join(f"{i}:{r[i]}" for i in ('a', 'b') if i in r) for r in rs)
            print(f"{mapping or 'a->x':9} {'attention' if attend else 'control  '}: {sum(r['ok'] for r in rs)}/{len(rs)}   {detail}")
