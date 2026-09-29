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

Temperature is per cell (a local thermostat): while being taught, a cell that nearly took part (lit to at least 15%
of its threshold, but didn't fire) and that attention neither reached nor passed within light's reach (two cells)
warms a little each trial,
up to a limit; a cell attention reaches cools back to the base; and all heat fades slowly. So chance firing
concentrates at near misses that go unrewarded, such as a closed gate beside a stopped pulse. (A cell beside a
rewarded route, such as a competing gate closing, must not warm; nor must a cell that fired, or heat spreads through
chance firing.) So chance
firing concentrates where activity keeps arriving and going nowhere (a closed gate beside a stopped pulse), and
stays low on routes that are working or idle.

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
from model import SimulationConfig, State

TRIAL = 10.0                 # seconds per trial: long enough for a pulse to cross
# cells within light's reach of a cell (two cells, as attention.FAR_NEIGHBOURS): attention passing this close to a cell
# means its surroundings are working, so its own silence is not a failure
LIGHT_REACH = np.array([[dx * dx + dy * dy <= 5 for dy in range(-2, 3)] for dx in range(-2, 3)])
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
    temperature: float = 0.05    # while being taught (with a thermostat: the base, to which rewarded cells cool)
    use_temperature: float = 0.0 # while being used (tested)
    heat: float = 0.01           # thermostat: warming per teaching trial of a lit cell attention didn't reach or pass (0: off)
    max_temperature: float = 0.12  # keep below the temperature at which every gate leaks (about 0.15)
    cool: float | None = None    # temperature a cell drops to when attention reaches it (None: the base temperature)
    relax: float = 0.05          # per teaching trial, every cell's temperature moves this fraction back to the base
    lit: float = 0.15            # "lit": received at least this fraction of its threshold light
    near_miss_only: bool = True  # warm only cells lit to `lit` of their threshold that didn't fire (not cells that fired)
    up: float = 0.2          # weight gained by a cause, per reward
    down: float = 0.05       # weight lost by a cell that fired beside the route without causing it, per reward
    tol: float = 0.3         # attention's cause tolerance (attention.causes)
    local: bool = False      # pass attention cell to cell (local_attention_step) instead of tracing it back (attention.relay)


FIRE_TRACE_TIME = 3.0        # seconds for a cell's fire trace to fade to 1/e


def shifted(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """b[x, y] = a[x + dx, y + dy]: each cell's view of its neighbour at offset (dx, dy); 0 beyond the edge."""
    b = np.zeros_like(a)
    w, h = a.shape
    b[max(0, -dx):min(w, w - dx), max(0, -dy):min(h, h - dy)] = a[max(0, dx):min(w, w + dx), max(0, dy):min(h, h + dy)]
    return b


def local_attention_step(trace: np.ndarray, attended: np.ndarray, suspect: np.ndarray, tol: float) -> None:
    """Attention passes one hop back, cell to cell, using only what each cell can see of its neighbours.

    Every cell shows its neighbours its fire trace (1 when it ignites, fading). An attended cell finds, among its
    neighbours that fired before it (smaller trace), the one that fired first; if none, among the cells two away,
    which light also reaches. It emits an acceptance level: a cause fired at least (1 - tol) of that neighbour's
    lead before it, i.e. its trace is at most trace^tol * earliest^(1 - tol). A neighbour in that ring whose own
    trace is at or below the level it receives becomes attended; a cell listens only to the attended neighbour that
    fired soonest after it, the one it could have caused. A cell that fired, has an
    attended neighbour and is not accepted is a suspect: if it is never accepted, it is a leak."""
    levels, earlier_exists = [], np.zeros(trace.shape, dtype=bool)
    for ring in (attention.NEIGHBOURS, attention.FAR_NEIGHBOURS):
        earliest = np.full(trace.shape, np.inf)
        for dx, dy in ring:
            n = shifted(trace, dx, dy)
            earliest = np.where((n > 0) & (n < trace), np.minimum(earliest, n), earliest)
        use = np.isfinite(earliest) & ~earlier_exists
        earlier_exists |= use
        level = np.where(attended & use, trace ** tol * np.where(use, earliest, 1.0) ** (1 - tol), 0.0)
        # each cell listens to the attended neighbour in this ring that fired soonest after it (the one it could have caused)
        next_after, received = np.full(trace.shape, np.inf), np.zeros(trace.shape)
        for dx, dy in ring:
            n, n_level = shifted(trace, dx, dy), shifted(level, dx, dy)
            closer = (n_level > 0) & (n > trace) & (n < next_after)
            next_after = np.where(closer, n, next_after)
            received = np.where(closer, n_level, received)
        levels.append(received)
    accept = ~attended & (trace > 0) & ((trace <= levels[0]) | (trace <= levels[1]))
    next_to_attended = np.zeros(trace.shape, dtype=bool)
    for dx, dy in attention.NEIGHBOURS:
        next_to_attended |= shifted(attended, dx, dy)
    suspect |= ~attended & ~accept & (trace > 0) & next_to_attended
    attended |= accept


class Run:
    """One sheet: adaptation (phase 1) by self.adapter, then trials of reward-driven learning, either whole (trial) or
    one time step at a time (begin, then advance until it returns the trial's result)."""

    def __init__(self, seed: int, teaching: Teaching | None = None, adapt_trials: int = 200):
        self.state, self.at, self.plastic, jitter = Layout(seed).build()
        self.adapter = attention.Trainer(self.state, self.at, self.plastic, attention.Rule(), [('a', 'o'), ('b', 'o')],
                                         jitter=jitter, seed=seed, outputs=('o',))
        self.config = broadcast_config()
        self.phase1 = [self.adapter.trial(self.config) for _ in range(adapt_trials)]
        self.teaching = teaching or Teaching()
        self.rng = np.random.default_rng(2000 + seed)
        self.dt = 1 / 50
        self.temperature = np.full(self.state.grid_size, self.teaching.temperature)

    def chance_ignitions(self, T) -> None:
        """Ignite some dark cells at random, more readily the nearer their light is to their threshold. T is one
        temperature or one per cell."""
        if np.max(T) <= 0:
            return
        s, p = self.state, self.config.vars
        ready = self.plastic & (s.flame_array == 0) & (s.energy_array >= p['STRIKE_LEVEL'])
        x = s.weight_array * s.illumination_array / p['MIN_STRIKE']
        rate = RATE_AT_THRESHOLD * np.exp(np.minimum(x - 1, 0) / np.maximum(T, 1e-6))
        fire = ready & (self.rng.random(s.grid_size) < -np.expm1(-rate * self.dt))
        s.flame_array[fire] = p['STRIKE_LEVEL']

    def trial(self, label: str | None, reward: bool) -> bool:
        """Strike label (or nothing), run for TRIAL seconds, and return whether o fired. With reward, if o fired,
        the teacher rewards what the network just did."""
        self.begin(label, reward)
        while (result := self.advance()) is None:
            pass
        return result['o']

    def begin(self, label: str | None, reward: bool) -> None:
        s = self.state
        s.reset_energy_and_flame()        # separate trials: nothing left over from the last one can pass for a response
        if label is not None:
            s.strike(self.at[label], self.config)
        self.rewarding = reward
        self.fired_at = np.full(s.grid_size, np.inf)
        self.peak = np.zeros(s.grid_size)
        self.steps = 0
        self.trace = np.zeros(s.grid_size)             # fire trace: 1 when a cell ignites, fading with FIRE_TRACE_TIME
        self.was_burning = np.zeros(s.grid_size, dtype=bool)
        self.attended = np.zeros(s.grid_size, dtype=bool)
        self.suspect = np.zeros(s.grid_size, dtype=bool)   # fired, next to an attended cell, not (yet) accepted as a cause
        self.seeded = False

    def advance(self) -> dict | None:
        """One time step of the current trial; at its end, the teacher's response and the trial's result."""
        s = self.state
        s.update(self.dt, self.config)
        self.chance_ignitions(self.temperature if self.rewarding else self.teaching.use_temperature)
        self.steps += 1
        self.fired_at[(s.flame_array > 0) & np.isinf(self.fired_at)] = self.steps * self.dt
        self.peak = np.maximum(self.peak, s.weight_array * s.illumination_array)
        burning = s.flame_array > 0
        self.trace *= np.exp(-self.dt / FIRE_TRACE_TIME)
        self.trace[burning & ~self.was_burning] = 1.0
        self.was_burning = burning
        if self.rewarding and self.teaching.local:
            o = self.at['o']
            first_fired_now = self.fired_at[o] == self.steps * self.dt
            if not self.seeded and first_fired_now and RESPONSE[0] <= self.fired_at[o] <= RESPONSE[1]:
                self.attended[o], self.seeded = True, True     # the teacher pays attention to o as it fires
            if self.seeded:
                local_attention_step(self.trace, self.attended, self.suspect, self.teaching.tol)
        if self.steps < round(TRIAL / self.dt):
            return None
        fired_at, t_o = self.fired_at, self.fired_at[self.at['o']]
        attended = np.zeros(s.grid_size, dtype=bool)
        if self.teaching.local:
            rewarded = self.seeded
            if rewarded:
                attended = self.attended.copy()
                self.apply(attended, self.suspect & ~attended & self.plastic)
        else:
            rewarded = self.rewarding and RESPONSE[0] <= t_o <= RESPONSE[1]
            if rewarded:
                attended = self.reward(fired_at)
        if self.rewarding:
            t, T = self.teaching, self.temperature
            T += t.relax * (t.temperature - T)
            if t.heat:
                near = self.peak >= t.lit * self.config.get('MIN_STRIKE')
                lit = self.plastic & (near & np.isinf(fired_at) if t.near_miss_only else near | np.isfinite(fired_at))
                ignored = lit & ~scipy.ndimage.binary_dilation(attended, LIGHT_REACH)
                T[ignored] = np.minimum(T[ignored] + t.heat, t.max_temperature)
            T[attended] = t.temperature if t.cool is None else t.cool
        return {'o': bool(np.isfinite(t_o)), 'rewarded': bool(rewarded), 'attended': int((attended & self.plastic).sum()),
                'active': float(np.isfinite(fired_at)[self.plastic].mean())}

    def reward(self, fired_at: np.ndarray) -> np.ndarray:
        """Strengthen the causes of o's firing, weaken the leaks beside them; return the cells attention reached."""
        t, w = self.teaching, self.state.weight_array
        attended = attention.relay(fired_at, self.at['o'], t.tol)
        beside = scipy.ndimage.binary_dilation(attended, np.ones((3, 3), dtype=bool)) & ~attended & self.plastic
        self.apply(attended, beside & np.isfinite(fired_at))
        return attended

    def apply(self, attended: np.ndarray, leaked: np.ndarray) -> None:
        """Causes (cells attention reached) strengthen; leaks (fired beside them without causing them) weaken."""
        t, w = self.teaching, self.state.weight_array
        up = attended & self.plastic
        w[up] = np.minimum(w[up] + t.up, self.config.get('WEIGHT_MAX'))
        w[leaked] = np.maximum(w[leaked] - t.down, self.config.get('WEIGHT_MIN'))

    def rates(self, trials: int = 10) -> dict:
        """How often o fires for a, for b and for no input, with no teacher."""
        return {k: sum(self.trial(None if k == '-' else k, False) for _ in range(trials)) / trials for k in ('a', 'b', '-')}


def score(seed: int, teaching: Teaching | None = None, teach_trials: int = 150, verbose: bool = False) -> dict:
    """Teach a, then b; after each, measure how often o fires for a, b and nothing."""
    run = Run(seed, teaching)
    log = lambda *a: verbose and print(*a, flush=True)
    result = {'seed': seed, 'adapted': run.rates(4)}
    log('after adaptation', result['adapted'])
    for on in ('a', 'b'):
        result[f'{on}_rewards'] = sum(run.trial(on, True) for _ in range(teach_trials))
        result[f'after_{on}'] = run.rates()
        log(f"after {teach_trials} {on}-trials ({result[f'{on}_rewards']} rewarded):", result[f'after_{on}'])
    ra, rb = result['after_a'], result['after_b']
    result['ok'] = ra['a'] >= 0.9 and ra['b'] <= 0.1 and rb['b'] >= 0.9 and rb['a'] <= 0.1
    return result


class Demo:
    """The whole experiment, one time step at a time, for the app (same interface as attention.Trainer): adaptation,
    then teaching a, then teaching b, with a test of each input (no teacher, no randomness) every test_every
    teaching trials. Each finished trial returns a summary with a line of text."""

    def __init__(self, seed: int, adapt_trials: int = 200, teach_trials: int = 150, test_every: int = 10):
        self.run = Run(seed, adapt_trials=0)
        self.state, self.at, self.plastic, self.rule = self.run.state, self.run.at, self.run.plastic, self.run.teaching
        plan = [('adapt', None)] * adapt_trials
        for on in ('a', 'b'):
            for n in range(1, teach_trials + 1):
                plan.append(('teach', on))
                if n % test_every == 0:
                    plan += [('test', 'a'), ('test', 'b')]
        self.plan = iter(plan)
        self.current = None
        self.trials = 0
        self.max_temperature = self.rule.max_temperature

    @property
    def temperature(self) -> np.ndarray:
        """Each sheet cell's teaching temperature, for the app's weight view (0 off the sheet)."""
        return np.where(self.plastic, self.run.temperature, 0.0)

    def step(self, dt: float, config: SimulationConfig) -> dict | None:
        if self.current is None:
            self.current = next(self.plan, ('done', None))
            if self.current[0] in ('teach', 'test'):
                self.run.begin(self.current[1], self.current[0] == 'teach')
        kind, label = self.current
        if kind == 'done':
            self.state.update(dt, config)
            return None
        if kind == 'adapt':
            r = self.run.adapter.step(dt, self.run.config)
            if r is None:
                return None
            text = f"adapting: {r['input']} -> o {'fired' if r['o'] else '-'}, active {r['active']:.0%}, attention reached {r['attended']} cells"
        else:
            r = self.run.advance()
            if r is None:
                return None
            if kind == 'teach':
                text = (f"teaching {label}: o {'fired, rewarded' if r['rewarded'] else 'fired, too late or early' if r['o'] else '-'}, "
                        f"hottest cell T {self.run.temperature[self.plastic].max():.2f}")
            else:
                text = f"   TEST {label} alone (no teacher, T 0): o {'FIRES' if r['o'] else 'silent'}"
        self.current = None
        self.trials += 1
        return {**r, 'text': text}


SETTINGS = {
    'attention traced back (experiment code)': Teaching(),
    'attention passed cell to cell (local)': Teaching(local=True),
}


if __name__ == '__main__':
    from multiprocessing import Pool
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    jobs = [(name, s) for name in SETTINGS for s in seeds]
    with Pool() as pool:
        results = pool.starmap(score, [(s, SETTINGS[name]) for name, s in jobs])
    fmt = lambda d: f"a{d['a']:.1f} b{d['b']:.1f} -{d['-']:.1f}"
    for name in SETTINGS:
        rs = [r for (n, _), r in zip(jobs, results) if n == name]
        print(f"{name}: ok {sum(r['ok'] for r in rs)}/{len(rs)}")
        for r in rs:
            print(f"   seed {r['seed']}: adapted [{fmt(r['adapted'])}]  after a [{fmt(r['after_a'])}]  after b [{fmt(r['after_b'])}]"
                  f"  rewards {r['a_rewards']},{r['b_rewards']}", flush=True)
