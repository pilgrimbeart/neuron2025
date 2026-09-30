"""Learning on a random sheet with local rules only: everything the cells do is physics.py.

The only things outside the cells are the training inputs and outputs:
  - striking input a or b (one pulse every TRIAL seconds; the grid runs continuously, never reset),
  - watching output o, and striking the teacher cell beside it when o responds (fires RESPONSE seconds after the
    input) on a trial where that response is wanted,
  - the `teaching` input, on while the network is being taught and off while it is being used.

Two stages, one rule: adaptation (both a and b rewarded for reaching o) grows routes from both inputs to o; then
teaching a (only a rewarded) and then teaching b. After each, a and b are tested with teaching off: the taught input
must reach o and the other must not.

    python twophase.py [SEEDS] [SIZE]      run on random SIZE x SIZE sheets (default 32) and report
    python twophase.py and [SEEDS]         teach AND instead (score_and)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import physics
from gates import hline, vline
from model import SimulationConfig, State

TRIAL = 12.0                 # seconds between input pulses on a 32-wide sheet: time to cross, respond, and recover
RESPONSE = (0.0, 9.0)        # o firing this long after the input (32-wide) counts as a response to it
TEACHER_DELAY = 0.3          # the teacher strikes its cell this long after seeing o respond
DT = 1 / 50


@dataclass
class Layout:
    """A random sheet of normal cells (weights START ± SPREAD) filling a grid of the given size, with inputs a and b,
    output o and GOOD and BAD teacher cells beside o. Margins and wires scale with the grid (at 32x32, input wires
    reach 6 cells into the sheet).

    Default: a (upper) and b (lower) enter from the left, and o leaves from the middle of the right edge.
    opposite: a enters from the left and b from the right, at the same height, and o leaves from the middle of the
    bottom edge, so routes from a and b meet at o's entry from opposite sides."""
    seed: int = 0
    size: tuple[int, int] = (32, 32)
    start: float = 0.45
    spread: float = 0.15
    opposite: bool = False

    def build(self) -> tuple[State, dict]:
        w, h = self.size
        entry = (3 * w) // 16       # the input wires continue this far into the sheet (a front rather than a point)
        if self.opposite:
            x0, y0, x1, y1 = w // 8, h // 8, w - 1 - w // 8, h - 1 - (3 * h) // 16
            row, col = (y0 + y1) // 2, w // 2
            wires = hline(row, 0, x0 - 1) | hline(row, x1 + 1, w - 1) | vline(col, y1 + 1, h - 1)
            fronts = [(slice(x0, x0 + entry), row), (slice(x1 - entry + 1, x1 + 1), row)]
            at = {'a': (0, row), 'b': (w - 1, row), 'o': (col, h - 1), 'good': (col + 1, h - 1), 'bad': (col - 1, h - 1)}
        else:
            x0, y0, x1, y1 = w // 8, h // 8, w - 1 - (3 * w) // 16, h - 1 - h // 8
            mid, top, bot = (y0 + y1) // 2, y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
            wires = hline(top, 0, x0 - 1) | hline(bot, 0, x0 - 1) | hline(mid, x1 + 1, w - 1)
            fronts = [(slice(x0, x0 + entry), top), (slice(x0, x0 + entry), bot)]
            at = {'a': (0, top), 'b': (0, bot), 'o': (w - 1, mid), 'good': (w - 1, mid + 1), 'bad': (w - 1, mid - 1)}
        s = State((w, h), seed=self.seed)
        sheet = {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
        for xy in sheet | wires:
            s.set_kind(xy, physics.NORMAL)
        s.set_kind(at['good'], physics.GOOD)
        s.set_kind(at['bad'], physics.BAD)
        rng = np.random.default_rng(self.seed)
        s.weight_array[x0:x1 + 1, y0:y1 + 1] = np.clip(self.start + rng.normal(0, self.spread, (x1 - x0 + 1, y1 - y0 + 1)), 0, None)
        for xs, y in fronts:
            s.weight_array[xs, y] = 1.0
        return s, at


class Experiment:
    """Runs trials one time step at a time (so the app can show them), or whole (trial). kind is 'teach' (teaching
    on: if o responds, the teacher rewards it when wanted, punishes it when not), or 'test' (teaching off, no
    teacher). label names the inputs struck together: 'a', 'b', 'ab', or None."""

    def __init__(self, seed: int = 0, size: tuple[int, int] = (32, 32), config: SimulationConfig | None = None,
                 start: float = 0.45, opposite: bool = False):
        self.state, self.at = Layout(seed, size, start, opposite=opposite).build()
        self.config = config or SimulationConfig()
        scale = size[0] / 32                    # a wider sheet takes proportionally longer to cross
        self.trial_time = TRIAL * scale
        self.response = (RESPONSE[0], RESPONSE[1] * scale)
        self.trials = 0
        self.plastic = self.state.kind_array == physics.NORMAL
        self.current = None

    def begin(self, kind: str, label: str | None, wanted: bool = True) -> None:
        self.current = (kind, label)
        self.wanted = wanted
        self.t = 0.0
        self.state.teaching = 1.0 if kind == 'teach' else 0.0
        for input_label in label or '':
            self.state.strike(self.at[input_label], self.config)
        self.responded = None
        self.reward_at = None
        self.o_burning = self.state.flame_array[self.at['o']] > 0

    def advance(self, dt: float = DT) -> dict | None:
        """One time step of the current trial; at its end, the trial's result."""
        s, o = self.state, self.at['o']
        s.update(dt, self.config)
        self.t += dt
        burning = s.flame_array[o] > 0
        if burning and not self.o_burning and self.responded is None and self.response[0] <= self.t <= self.response[1]:
            self.responded = self.t
            if self.current[0] == 'teach':
                self.reward_at = self.t + TEACHER_DELAY
                self.teacher = 'good' if self.wanted else 'bad'
        self.o_burning = burning
        if self.reward_at is not None and self.t >= self.reward_at:
            s.strike(self.at[self.teacher], self.config)
            self.reward_at = None
        if self.t < self.trial_time - dt / 2:
            return None
        self.trials += 1
        kind, label = self.current
        self.current = None
        taught = kind == 'teach' and self.responded is not None
        return {'kind': kind, 'input': label, 'o': self.responded is not None,
                'rewarded': taught and self.wanted, 'punished': taught and not self.wanted}

    def trial(self, kind: str, label: str | None, wanted: bool = True) -> dict:
        self.begin(kind, label, wanted)
        while (result := self.advance()) is None:
            pass
        return result

    def rates(self, trials: int = 10, labels: tuple = ('a', 'b', '-')) -> dict:
        """How often o responds to each of labels ('-' is no input), with teaching off."""
        return {k: sum(self.trial('test', None if k == '-' else k)['o'] for _ in range(trials)) / trials
                for k in labels}


def score(seed: int, adapt_trials: int = 200, teach_trials: int = 150, verbose: bool = False,
          params: dict | None = None, size: tuple[int, int] = (32, 32), start: float = 0.45) -> dict:
    """Adapt (a and b both rewarded), then teach a, then teach b; measure after each stage. params overrides some of
    physics.DEFAULT_PARAMS (for experiments)."""
    ex = Experiment(seed, size, SimulationConfig(dict(physics.DEFAULT_PARAMS, **(params or {}))), start)
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


def score_and(seed: int, adapt_trials: int = 200, and_trials: int = 300, verbose: bool = False,
              params: dict | None = None, size: tuple[int, int] = (32, 32), start: float = 0.45,
              opposite: bool = False, block_trials: int = 0, shuffle: bool = False) -> dict:
    """Adapt (a and b both rewarded; adapt_trials=0 skips this), then teach AND: a and b struck together are rewarded
    if o responds, a alone or b alone punished if it does, in rotation or (shuffle) in random order. With teaching off,
    a+b must reach o and neither alone may."""
    ex = Experiment(seed, size, SimulationConfig(dict(physics.DEFAULT_PARAMS, **(params or {}))), start, opposite)
    log = lambda *a: verbose and print(*a, flush=True)
    labels = ('ab', 'a', 'b', '-')
    for n in range(adapt_trials):
        ex.trial('teach', 'ab'[n % 2])
    result = {'seed': seed, 'adapted': ex.rates(4, labels)}
    log('after adaptation', result['adapted'])
    for n in range(block_trials):               # first, punish single inputs only, until neither passes alone
        ex.trial('teach', 'ab'[n % 2], False)
    if block_trials:
        result['blocked'] = ex.rates(4, labels)
        log('after blocking', result['blocked'])
    schedule = [('ab', True), ('a', False), ('b', False)]
    rewards = punishments = 0
    order = np.random.default_rng(3000 + seed)               # shuffle: each trial's input chosen at random
    for n in range(and_trials):
        r = ex.trial('teach', *schedule[order.integers(3) if shuffle else n % 3])
        rewards += r['rewarded']
        punishments += r['punished']
        if verbose and n % 30 == 29:
            log(f'  after {n + 1} AND trials ({rewards} rewarded, {punishments} punished):', ex.rates(2, labels))
    result.update(rewards=rewards, punishments=punishments, after=ex.rates(10, labels))
    a = result['after']
    result['ok'] = a['ab'] >= 0.9 and a['a'] <= 0.1 and a['b'] <= 0.1
    return result


class Demo:
    """The whole experiment, one time step at a time, for the app: adaptation, then teaching a, then teaching b,
    with a test of each input every test_every teaching trials. Each finished trial returns a line of text."""

    def __init__(self, seed: int, size: tuple[int, int] = (32, 32), config: SimulationConfig | None = None,
                 adapt_trials: int = 200, teach_trials: int = 150, test_every: int = 10):
        self.ex = Experiment(seed, size, config)
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
    if sys.argv[1:2] == ['and']:
        seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
        with Pool() as pool:
            results = pool.map(score_and, seeds)
        fmt = lambda d: ' '.join(f"{k}{v:.1f}" for k, v in d.items())
        print(f"AND: ok {sum(r['ok'] for r in results)}/{len(results)}")
        for r in results:
            print(f"   seed {r['seed']}: adapted [{fmt(r['adapted'])}]  after [{fmt(r['after'])}]  "
                  f"rewarded {r['rewards']}, punished {r['punishments']}", flush=True)
        sys.exit()
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    with Pool() as pool:
        results = pool.starmap(score, [(seed, 200, 150, False, None, (n, n)) for seed in seeds])
    fmt = lambda d: f"a{d['a']:.1f} b{d['b']:.1f} -{d['-']:.1f}"
    print(f"ok {sum(r['ok'] for r in results)}/{len(results)}")
    for r in results:
        print(f"   seed {r['seed']}: adapted [{fmt(r['adapted'])}]  after a [{fmt(r['after_a'])}]  after b [{fmt(r['after_b'])}]"
              f"  rewards {r['a_rewards']},{r['b_rewards']}", flush=True)
