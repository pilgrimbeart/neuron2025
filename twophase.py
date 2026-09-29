"""Learning on a random sheet with local rules only: everything the cells do is physics.py.

The only things outside the cells are the training inputs and outputs:
  - striking input a or b (one pulse every TRIAL seconds; the grid runs continuously, never reset),
  - watching output o, and striking the teacher cell beside it when o responds (fires RESPONSE seconds after the
    input) on a trial where that response is wanted,
  - the `teaching` input, on while the network is being taught and off while it is being used.

Two stages, one rule: adaptation (both a and b rewarded for reaching o) grows routes from both inputs to o; then
teaching a (only a rewarded) and then teaching b. After each, a and b are tested with teaching off: the taught input
must reach o and the other must not.

    python twophase.py [SEEDS]      run on random sheets and report
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import physics
from gates import hline
from model import SimulationConfig, State

TRIAL = 12.0                 # seconds between input pulses: time to cross, respond, and recover
RESPONSE = (1.0, 9.0)        # o firing this long after the input counts as a response to it
TEACHER_DELAY = 0.3          # the teacher strikes its cell this long after seeing o respond
DT = 1 / 50


@dataclass
class Layout:
    """A random sheet of normal cells (weights START ± SPREAD) with inputs a (upper) and b (lower) entering from the
    left, output o leaving from the middle of the right edge, and a teacher cell beside o."""
    seed: int = 0
    size: int = 32
    box: tuple[int, int, int, int] = (4, 4, 25, 27)
    start: float = 0.45
    spread: float = 0.15
    entry: int = 6           # the input wires continue this far into the sheet (a front rather than a point)

    def build(self) -> tuple[State, dict]:
        x0, y0, x1, y1 = self.box
        n = self.size - 1
        mid, top, bot = (y0 + y1) // 2, y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        s = State((self.size, self.size), seed=self.seed)
        sheet = {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
        for xy in sheet | hline(top, 0, x0 - 1) | hline(bot, 0, x0 - 1) | hline(mid, x1 + 1, n):
            s.set_kind(xy, physics.NORMAL)
        s.set_kind((n, mid + 1), physics.TEACHER)
        rng = np.random.default_rng(self.seed)
        s.weight_array[x0:x1 + 1, y0:y1 + 1] = np.clip(self.start + rng.normal(0, self.spread, (x1 - x0 + 1, y1 - y0 + 1)), 0, None)
        for row in (top, bot):
            s.weight_array[x0:x0 + self.entry, row] = 1.0
        return s, {'a': (0, top), 'b': (0, bot), 'o': (n, mid), 'teacher': (n, mid + 1)}


class Experiment:
    """Runs trials one time step at a time (so the app can show them), or whole (trial). kind is 'teach' (teaching
    on, the teacher rewards o responding to label), or 'test' (teaching off, no teacher)."""

    def __init__(self, seed: int = 0, config: SimulationConfig | None = None):
        self.state, self.at = Layout(seed).build()
        self.config = config or SimulationConfig()
        self.trials = 0
        self.plastic = self.state.kind_array == physics.NORMAL
        self.current = None

    def begin(self, kind: str, label: str | None) -> None:
        self.current = (kind, label)
        self.t = 0.0
        self.state.teaching = 1.0 if kind == 'teach' else 0.0
        if label is not None:
            self.state.strike(self.at[label], self.config)
        self.responded = None
        self.reward_at = None
        self.o_burning = self.state.flame_array[self.at['o']] > 0

    def advance(self, dt: float = DT) -> dict | None:
        """One time step of the current trial; at its end, the trial's result."""
        s, o = self.state, self.at['o']
        s.update(dt, self.config)
        self.t += dt
        burning = s.flame_array[o] > 0
        if burning and not self.o_burning and self.responded is None and RESPONSE[0] <= self.t <= RESPONSE[1]:
            self.responded = self.t
            if self.current[0] == 'teach':
                self.reward_at = self.t + TEACHER_DELAY
        self.o_burning = burning
        if self.reward_at is not None and self.t >= self.reward_at:
            s.strike(self.at['teacher'], self.config)
            self.reward_at = None
        if self.t < TRIAL - dt / 2:
            return None
        self.trials += 1
        kind, label = self.current
        self.current = None
        return {'kind': kind, 'input': label, 'o': self.responded is not None,
                'rewarded': kind == 'teach' and self.responded is not None}

    def trial(self, kind: str, label: str | None) -> dict:
        self.begin(kind, label)
        while (result := self.advance()) is None:
            pass
        return result

    def rates(self, trials: int = 10) -> dict:
        """How often o responds to a, to b and to no input, with teaching off."""
        return {k: sum(self.trial('test', None if k == '-' else k)['o'] for _ in range(trials)) / trials
                for k in ('a', 'b', '-')}


def score(seed: int, adapt_trials: int = 200, teach_trials: int = 150, verbose: bool = False,
          params: dict | None = None) -> dict:
    """Adapt (a and b both rewarded), then teach a, then teach b; measure after each stage. params overrides some of
    physics.DEFAULT_PARAMS (for experiments)."""
    ex = Experiment(seed, SimulationConfig(dict(physics.DEFAULT_PARAMS, **(params or {}))))
    log = lambda *a: verbose and print(*a, flush=True)
    for n in range(adapt_trials):
        ex.trial('teach', 'ab'[n % 2])
    result = {'seed': seed, 'adapted': ex.rates(4)}
    log('after adaptation', result['adapted'])
    for on in ('a', 'b'):
        result[f'{on}_rewards'] = sum(ex.trial('teach', on)['rewarded'] for _ in range(teach_trials))
        result[f'after_{on}'] = ex.rates()
        log(f"after teaching {on} ({result[f'{on}_rewards']} rewarded):", result[f'after_{on}'])
    ra, rb = result['after_a'], result['after_b']
    result['ok'] = ra['a'] >= 0.9 and ra['b'] <= 0.1 and rb['b'] >= 0.9 and rb['a'] <= 0.1
    return result


class Demo:
    """The whole experiment, one time step at a time, for the app: adaptation, then teaching a, then teaching b,
    with a test of each input every test_every teaching trials. Each finished trial returns a line of text."""

    def __init__(self, seed: int, adapt_trials: int = 200, teach_trials: int = 150, test_every: int = 10):
        self.ex = Experiment(seed)
        self.state, self.at, self.plastic = self.ex.state, self.ex.at, self.ex.plastic
        self.max_temperature = self.ex.config.get('T_MAX')
        plan = [('teach', 'ab'[n % 2], 'adapting') for n in range(adapt_trials)]
        for on in ('a', 'b'):
            for n in range(1, teach_trials + 1):
                plan.append(('teach', on, f'teaching {on}'))
                if n % test_every == 0:
                    plan += [('test', 'a', ''), ('test', 'b', '')]
        self.plan = iter(plan)
        self.stage = ''
        self.trials = 0

    @property
    def temperature(self) -> np.ndarray:
        """Each cell's chance-firing temperature, for the app's weight view (0 where there is no cell)."""
        return np.where(self.plastic, self.ex.config.get('T_BASE') + self.state.heat_array, 0.0)

    def step(self, dt: float, config: SimulationConfig) -> dict | None:
        if self.ex.current is None:
            kind, label, self.stage = next(self.plan, ('test', None, 'finished'))
            self.ex.begin(kind, label)
        r = self.ex.advance(dt)
        if r is None:
            return None
        self.trials += 1
        if r['kind'] == 'test' and r['input'] is None:
            return None
        if r['kind'] == 'test':
            text = f"   TEST {r['input']} alone (teaching off): o {'RESPONDS' if r['o'] else 'silent'}"
        else:
            text = f"{self.stage}: {r['input']} -> o {'responded, rewarded' if r['rewarded'] else '-'}"
        hottest = self.temperature[self.plastic].max()
        return {**r, 'text': f"{text}, hottest cell T {hottest:.2f}"}


if __name__ == '__main__':
    from multiprocessing import Pool
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    with Pool() as pool:
        results = pool.map(score, seeds)
    fmt = lambda d: f"a{d['a']:.1f} b{d['b']:.1f} -{d['-']:.1f}"
    print(f"ok {sum(r['ok'] for r in results)}/{len(results)}")
    for r in results:
        print(f"   seed {r['seed']}: adapted [{fmt(r['adapted'])}]  after a [{fmt(r['after_a'])}]  after b [{fmt(r['after_b'])}]"
              f"  rewards {r['a_rewards']},{r['b_rewards']}", flush=True)
