"""Learning experiments: test circuits, candidate learning rules, and a scripted protocol that scores them.

    python learn.py            run the protocol on T1 with rule M1 and print the trace

Rules live here, applied after each physics step, until one wins and moves into physics.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import physics
from gates import hline, vline
from model import SimulationConfig, State

DT = 1 / 50
GAP = 10.0          # seconds of quiet after every stimulus: longer than the refractory period


# --- test circuits ------------------------------------------------------------------------------------

def t1() -> tuple[State, dict[str, tuple[int, int]]]:
    """One junction J=(16,16). a's line ends diagonally beside J, so a alone falls short of igniting it.
    J starts the output line. l's line ends in a transducer two cells below J: close enough to teach J,
    far enough that J's own light can't ignite it (light and teaching signal reach are reciprocal)."""
    s = State((32, 32))
    for xy in hline(15, 2, 15) | hline(16, 16, 29) | vline(16, 19, 31):
        s.set_kind(xy, physics.NORMAL)
    s.set_kind((16, 18), physics.TRANSDUCER)
    return s, {'a': (2, 15), 'l': (16, 31), 'j': (16, 16), 'o': (29, 16)}


def t2() -> tuple[State, dict[str, tuple[int, int]]]:
    """Two T1-style junctions sharing one l and one output. a (left) and b (right) each end diagonally
    beside their junction; l splits into a transducer diagonally beside each. Each junction's output goes up
    through a diode into an OR relay, and out at the top edge. All three input paths are 18 cells long."""
    from gates import diode
    half = hline(21, 0, 6) | vline(6, 15, 21) | hline(15, 6, 12)              # a's line, ending at (12,15)
    half |= vline(13, 8, 13) | {(13, 14)}                                     # J_A (13,14) and its output
    half |= diode((13, 6), (0, -1)) | vline(13, 4, 6) | hline(4, 13, 16)      # diode up into the relay
    half |= {(15, 15)}                                                        # l's left branch
    cells = half | {(32 - x, y) for x, y in half} | vline(16, 15, 31) | vline(16, 0, 4)
    s = State((33, 33))
    for xy in cells:
        s.set_kind(xy, physics.NORMAL)
    for xy in [(15, 14), (17, 14)]:                                           # two cells from each junction
        s.set_kind(xy, physics.TRANSDUCER)
    return s, {'a': (0, 21), 'b': (32, 21), 'l': (16, 31), 'o': (16, 0), 'ja': (13, 14), 'jb': (19, 14)}


def show(circuit) -> None:
    s, at = circuit()
    g = np.full(s.grid_size, '.')
    g[s.kind_array == physics.NORMAL] = '#'
    g[s.kind_array == physics.TRANSDUCER] = 'T'
    for label, xy in at.items():
        g[xy] = label[0].upper() if len(label) == 1 else label[1].upper()
    print('\n'.join(''.join(g[x, y] for x in range(s.grid_size[0])) for y in range(s.grid_size[1])))


# --- learning rules ------------------------------------------------------------------------------------

def eligible(s: State, mode: str) -> np.ndarray:
    cells = s.kind_array != physics.EMPTY
    if mode in ('dark', 'ready'):
        cells &= s.flame_array == 0
    if mode == 'ready':
        cells &= s.energy_array >= physics.DEFAULT_PARAMS['STRIKE_LEVEL']
    return cells


@dataclass
class M1:
    """The receiver learns to listen: a cell's weight changes at rate*signal*(light - offset), within bounds.
    With dark_only, only cells that aren't burning learn (so learning stops once the cell can fire)."""
    rate: float = 48.0
    offset: float = 0.02
    low: float = 0.5
    high: float = 2.0
    eligible: str = 'all'

    def apply(self, s: State, dt: float) -> None:
        cells = eligible(s, self.eligible)
        change = self.rate * s.modulator_array * (s.illumination_array - self.offset) * dt
        s.weight_array[cells] = np.clip(s.weight_array[cells] + change[cells], self.low, self.high)


@dataclass
class M1b:
    """Receiver learns to listen, split rule: under teaching signal, a lit cell (light > dark) strengthens in
    proportion to its light; a dark cell weakens at a fixed rate. Within bounds."""
    rate: float = 48.0
    unlearn: float = 2.4
    dark: float = 0.06
    low: float = 0.5
    high: float = 2.0
    eligible: str = 'all'

    def apply(self, s: State, dt: float) -> None:
        cells = eligible(s, self.eligible)
        lit = s.illumination_array > self.dark
        change = s.modulator_array * np.where(lit, self.rate * s.illumination_array, -self.unlearn) * dt
        s.weight_array[cells] = np.clip(s.weight_array[cells] + change[cells], self.low, self.high)


@dataclass
class M1c:
    """Receiver learns to listen, scale-free: under teaching signal M, weight changes at M*(rate*light - unlearn*M).
    Lighter than the teaching signal (by rate/unlearn) strengthens; darker weakens. Both terms scale alike with gain."""
    rate: float = 48.0
    unlearn: float = 72.0
    low: float = 0.5
    high: float = 2.0
    eligible: str = 'ready'      # 'all' cells, 'dark' cells, or 'ready' cells (dark and able to ignite)

    def apply(self, s: State, dt: float) -> None:
        cells = eligible(s, self.eligible)
        m = s.modulator_array
        change = m * (self.rate * s.illumination_array - self.unlearn * m) * dt
        s.weight_array[cells] = np.clip(s.weight_array[cells] + change[cells], self.low, self.high)


@dataclass
class M1d:
    """Receiver learns to listen. Under teaching signal M: any cell strengthens at rate*M*light (so a burning
    junction keeps consolidating); only ready cells (dark and able to ignite) weaken at unlearn*M^2."""
    rate: float = 48.0
    unlearn: float = 72.0
    low: float = 0.5
    high: float = 2.0

    def apply(self, s: State, dt: float) -> None:
        m = s.modulator_array
        up = self.rate * m * s.illumination_array * (s.kind_array != physics.EMPTY)
        down = self.unlearn * m * m * eligible(s, 'ready')
        s.weight_array[:] = np.clip(s.weight_array + (up - down) * dt, self.low, self.high)


# --- protocol ------------------------------------------------------------------------------------------

class Run:
    def __init__(self, circuit, rule, gain=1.0, fps=50):
        self.state, self.at = circuit()
        self.rule = rule
        self.config = SimulationConfig(dict(physics.DEFAULT_PARAMS))
        self.config.set('COUPLING_GAIN', self.config.get('COUPLING_GAIN') * gain)
        self.dt = 1 / fps

    def pulse(self, *labels: str) -> dict[str, int]:
        """Strike labels together, run GAP seconds, and return how often each probe ignited."""
        s = self.state
        counts = {k: 0 for k in self.at}
        for label in labels:
            s.strike(self.at[label], self.config)
        for _ in range(round(GAP / self.dt)):
            before = s.flame_array > 0
            s.update(self.dt, self.config)
            self.rule.apply(s, self.dt)
            lit = (s.flame_array > 0) & ~before
            for k, xy in self.at.items():
                counts[k] += int(lit[xy])
        return counts

    def passes(self, label: str) -> bool:
        return self.pulse(label)['o'] > 0


def protocol_t1(rule, gain=1.0, fps=50, pairings=10, verbose=False) -> dict:
    run = Run(t1, rule, gain, fps)
    log = lambda *a: verbose and print(*a)
    result = {'before': run.passes('a')}
    log(f'before training: a -> o {result["before"]}   w_J={run.state.weight_array[run.at["j"]]:.3f}')
    learned_at = None
    for n in range(1, pairings + 1):
        paired = run.pulse('a', 'l')['o'] > 0
        ok = run.passes('a')
        log(f'pairing {n}: o fired during pairing {paired}, then a alone -> o {ok}   w_J={run.state.weight_array[run.at["j"]]:.3f}')
        if ok and learned_at is None:
            learned_at = n
    result['learned_at'] = learned_at
    result['retained_use'] = all(run.passes('a') for _ in range(10))
    result['l_alone_silent'] = not run.passes('l')
    log(f'after 10 more uses of a: still passes {result["retained_use"]};  l alone silent {result["l_alone_silent"]}   w_J={run.state.weight_array[run.at["j"]]:.3f}')
    for n in range(1, 11):
        run.pulse('l')
        if not run.passes('a'):
            result['untrained_after'] = n
            break
    log(f'untrained by l alone after {result.get("untrained_after")} pulses   w_J={run.state.weight_array[run.at["j"]]:.3f}')
    return result



def protocol_t2(rule, gain=1.0, fps=50, pairings=12, verbose=False) -> dict:
    """Pair a+l a fixed number of times, noting when a first passes (and b doesn't); then pair b+l the same
    number of times, noting when it first switches; then check that b passes and a doesn't, repeatedly,
    and that l alone never reaches the output."""
    run = Run(t2, rule, gain, fps)
    w = lambda: f"w_A={run.state.weight_array[run.at['ja']]:.2f} w_B={run.state.weight_array[run.at['jb']]:.2f}"
    log = lambda *a: verbose and print(*a)
    result = {'before_silent': not run.passes('a') and not run.passes('b')}
    log(f"before: a and b silent {result['before_silent']}   {w()}")
    for phase, (on, off) in (('a_first_at', ('a', 'b')), ('b_first_at', ('b', 'a'))):
        result[phase] = None
        for n in range(1, pairings + 1):
            run.pulse(on, 'l')
            ok = run.passes(on) and not run.passes(off)
            log(f"{on}+l pairing {n}: {on} alone passes and {off} doesn't: {ok}   {w()}")
            if ok and result[phase] is None:
                result[phase] = n
        if phase == 'a_first_at':
            result['a_reliable'] = all(run.passes('a') and not run.passes('b') for _ in range(5))
    result['b_reliable'] = all(run.passes('b') and not run.passes('a') for _ in range(5))
    result['l_alone_silent'] = not run.passes('l')
    log(f"b reliable {result['b_reliable']}, l alone silent {result['l_alone_silent']}   {w()}")
    return result

if __name__ == '__main__':
    print(protocol_t1(M1(), verbose=True))
