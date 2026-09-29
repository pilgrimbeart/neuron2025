"""Two loops on one random sheet: slow adaptation grows routes, then reward-driven learning with a temperature
selects between them.

Phase 1, adaptation (attention.py's rule, deterministic): trials alternate inputs a and b, and attention is paid to
output o whenever it fires. This grows routes from both inputs to o.

Phase 2, learning: the network acts by itself and the teacher only rewards what it just did. Each trial strikes one
input; if o fires, the teacher says "yes": attention runs back from o through the causes of its firing (as in
attention.py). Cells attention reaches strengthen; cells beside the route that fired without causing it (backflow
into a competing route, or a side leak) weaken. A cell that didn't fire did nothing, so it is neither rewarded nor
blamed. With no reward nothing changes.

Exploration comes from temperature: a cell at rest ignites by chance, at a rate that rises steeply as its light
approaches its threshold (rate = RATE_AT_THRESHOLD * exp((x - 1) / T), with x = weight * light / MIN_STRIKE, and
certain ignition from x = 1 as usual). T = 0 is physics.py's deterministic rule. So a gate that has been closed just
below its threshold occasionally lets a pulse through, and the teacher can reward it.

The test: teach a (reward o on a-trials); then a must pass and b must not. Then teach b; then b must pass and a
must not. Measured throughout: how often o fires for each input and with no input at all.

    python twophase.py [SEEDS]      sweep the temperature on random sheets and report
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import scipy.ndimage

import attention
import physics
from broadcast import broadcast_config
from gates import hline
from model import State

TRIAL = 10.0                 # seconds per trial: long enough for a pulse to cross
RESPONSE = (1.0, 6.0)        # the teacher only rewards o firing this many seconds after the input: a response to it
RATE_AT_THRESHOLD = 5.0      # chance ignitions per second of a cell whose light just reaches its threshold


@dataclass
class Layout:
    """A random sheet with inputs a (upper) and b (lower) entering from the left, and output o leaving from the
    middle of the right edge."""
    seed: int = 0
    size: int = 32
    box: tuple[int, int, int, int] = (4, 4, 25, 27)
    spread: float = 0.15
    entry: int = 6

    def build(self):
        x0, y0, x1, y1 = self.box
        n = self.size - 1
        mid, top, bot = (y0 + y1) // 2, y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        s = State((self.size, self.size))
        sheet = {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
        for xy in sheet | hline(top, 0, x0 - 1) | hline(bot, 0, x0 - 1) | hline(mid, x1 + 1, n):
            s.set_kind(xy, physics.NORMAL)
        plastic = np.zeros(s.grid_size, dtype=bool)
        plastic[x0:x1 + 1, y0:y1 + 1] = True
        for row in (top, bot):
            plastic[x0:x0 + self.entry, row] = False          # the input wires continue a little into the sheet, fixed
            s.weight_array[x0:x0 + self.entry, row] = 1.0
        plastic[x1 + 1, mid] = True                            # o's synapse onto the sheet
        jitter = np.random.default_rng(self.seed).normal(0, self.spread, s.grid_size) * plastic
        return s, {'a': (0, top), 'b': (0, bot), 'o': (n, mid), 'entry': (x1 + 1, mid)}, plastic, jitter


@dataclass
class Teaching:
    temperature: float = 0.1     # while being taught
    use_temperature: float = 0.0 # while being used (tested)
    up: float = 0.2          # weight gained by a cause, per reward
    down: float = 0.05       # weight lost by a cell that fired beside the route without causing it, per reward
    tol: float = 0.3         # attention's cause tolerance (attention.causes)


class Run:
    def __init__(self, seed: int, teaching: Teaching | None = None, adapt_trials: int = 200):
        self.state, self.at, self.plastic, jitter = Layout(seed).build()
        trainer = attention.Trainer(self.state, self.at, self.plastic, attention.Rule(), [('a', 'o'), ('b', 'o')],
                                    jitter=jitter, seed=seed, outputs=('o',))
        self.config = broadcast_config()
        self.phase1 = [trainer.trial(self.config) for _ in range(adapt_trials)]
        self.teaching = teaching or Teaching()
        self.rng = np.random.default_rng(2000 + seed)
        self.dt = 1 / 50

    def chance_ignitions(self, T: float) -> None:
        """Ignite some dark cells at random, more readily the nearer their light is to their threshold."""
        if T <= 0:
            return
        s, p = self.state, self.config.vars
        ready = self.plastic & (s.flame_array == 0) & (s.energy_array >= p['STRIKE_LEVEL'])
        x = s.weight_array * s.illumination_array / p['MIN_STRIKE']
        rate = RATE_AT_THRESHOLD * np.exp(np.minimum(x - 1, 0) / T)
        fire = ready & (self.rng.random(s.grid_size) < -np.expm1(-rate * self.dt))
        s.flame_array[fire] = p['STRIKE_LEVEL']

    def trial(self, label: str | None, reward: bool) -> bool:
        """Strike label (or nothing), run for TRIAL seconds, and return whether o fired. With reward, if o fired,
        the teacher rewards what the network just did."""
        s = self.state
        T = self.teaching.temperature if reward else self.teaching.use_temperature
        s.reset_energy_and_flame()        # separate trials: nothing left over from the last one can pass for a response
        if label is not None:
            s.strike(self.at[label], self.config)
        fired_at = np.full(s.grid_size, np.inf)
        for step in range(round(TRIAL / self.dt)):
            s.update(self.dt, self.config)
            self.chance_ignitions(T)
            fired_at[(s.flame_array > 0) & np.isinf(fired_at)] = (step + 1) * self.dt
        t_o = fired_at[self.at['o']]
        if reward and RESPONSE[0] <= t_o <= RESPONSE[1]:
            self.reward(fired_at)
        return bool(np.isfinite(t_o))

    def reward(self, fired_at: np.ndarray) -> None:
        t, w = self.teaching, self.state.weight_array
        attended = attention.relay(fired_at, self.at['o'], t.tol)
        beside = scipy.ndimage.binary_dilation(attended, np.ones((3, 3), dtype=bool)) & ~attended & self.plastic
        leaked = beside & np.isfinite(fired_at)
        up = attended & self.plastic
        w[up] = np.minimum(w[up] + t.up, self.config.get('WEIGHT_MAX'))
        w[leaked] = np.maximum(w[leaked] - t.down, self.config.get('WEIGHT_MIN'))

    def rates(self, trials: int = 10) -> dict:
        """How often o fires for a, for b and for no input, with no teacher."""
        return {k: sum(self.trial(None if k == '-' else k, False) for _ in range(trials)) / trials for k in ('a', 'b', '-')}


def score(seed: int, temperature: float = 0.1, teach_trials: int = 60, verbose: bool = False, use_temperature: float = 0.0) -> dict:
    """Teach a, then b; after each, measure how often o fires for a, b and nothing."""
    run = Run(seed, Teaching(temperature=temperature, use_temperature=use_temperature))
    log = lambda *a: verbose and print(*a, flush=True)
    result = {'seed': seed, 'T': temperature, 'adapted': run.rates(4)}
    log('after adaptation', result['adapted'])
    for on in ('a', 'b'):
        result[f'{on}_rewards'] = sum(run.trial(on, True) for _ in range(teach_trials))
        result[f'after_{on}'] = run.rates()
        log(f"after {teach_trials} {on}-trials ({result[f'{on}_rewards']} rewarded):", result[f'after_{on}'])
    ra, rb = result['after_a'], result['after_b']
    result['ok'] = ra['a'] >= 0.9 and ra['b'] <= 0.1 and rb['b'] >= 0.9 and rb['a'] <= 0.1
    return result


if __name__ == '__main__':
    from multiprocessing import Pool
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    temperatures = (0.1, 0.12, 0.15)
    jobs = [(s, T) for T in temperatures for s in seeds]
    with Pool() as pool:
        results = pool.starmap(score, [(s, T, 150) for s, T in jobs])
    fmt = lambda d: f"a{d['a']:.1f} b{d['b']:.1f} -{d['-']:.1f}"
    for T in temperatures:
        rs = [r for r in results if r['T'] == T]
        print(f"T {T}: ok {sum(r['ok'] for r in rs)}/{len(rs)}")
        for r in rs:
            print(f"   seed {r['seed']}: adapted [{fmt(r['adapted'])}]  after a [{fmt(r['after_a'])}]  after b [{fmt(r['after_b'])}]"
                  f"  rewards {r['a_rewards']},{r['b_rewards']}", flush=True)
