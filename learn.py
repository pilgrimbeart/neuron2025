"""Training protocols for the learning patterns (learn1, learn2 in gates.py), run headlessly.

    python learn.py            train learn1 and learn2 at nominal settings and print what happens

The learning rule itself is step 7 of physics.py. verify.py runs these protocols with margins.
"""

from __future__ import annotations

import sys

import gates
import physics
from model import SimulationConfig

GAP = 15.0          # seconds after every stimulus: long enough for the slowest pulse to arrive and the grid to recover


class Run:
    def __init__(self, name: str, gain: float = 1.0, fps: int = 50):
        self.state, probes = gates.build_state(name)
        self.at = {p['label']: p['xy'] for p in probes}
        self.config = SimulationConfig(dict(physics.DEFAULT_PARAMS))
        self.config.set('COUPLING_GAIN', self.config.get('COUPLING_GAIN') * gain)
        self.dt = 1 / fps

    def pulse(self, *labels: str) -> bool:
        """Strike labels together, run GAP seconds, and return whether the output o ignited."""
        s, o = self.state, self.at['o']
        for label in labels:
            s.strike(self.at[label], self.config)
        fired = False
        for _ in range(round(GAP / self.dt)):
            was_burning = s.flame_array[o] > 0
            s.update(self.dt, self.config)
            fired |= bool(s.flame_array[o] > 0 and not was_burning)
        return fired

    def weights(self, *labels: str) -> str:
        return ' '.join(f"w_{k}={self.state.weight_array[self.at[k]]:.2f}" for k in labels)


def train_learn1(gain=1.0, fps=50, pairings=12, verbose=False) -> dict:
    """a alone must not pass before training, must pass after pairing a+l, keep passing through use, never
    pass from l alone, and be untaught by l alone."""
    run = Run('learn1', gain, fps)
    log = lambda *a: verbose and print(*a)
    result = {'before_silent': not run.pulse('a'), 'first_at': None}
    for n in range(1, pairings + 1):
        run.pulse('a', 'l')
        passes = run.pulse('a')
        log(f'a+l pairing {n}: a alone passes {passes}   {run.weights("j")}')
        if passes and result['first_at'] is None:
            result['first_at'] = n
    result['reliable'] = all(run.pulse('a') for _ in range(5))
    result['l_alone_silent'] = not run.pulse('l')
    result['untaught'] = False
    for _ in range(8):
        run.pulse('l')
        if not run.pulse('a'):
            result['untaught'] = True
            break
    log(f"reliable {result['reliable']}, l alone silent {result['l_alone_silent']}, untaught by l alone {result['untaught']}   {run.weights('j')}")
    result['ok'] = result['before_silent'] and result['reliable'] and result['l_alone_silent'] and result['untaught']
    return result


def train_learn2(gain=1.0, fps=50, pairings=12, verbose=False) -> dict:
    """Before training neither a nor b passes. After pairing a+l, a passes and b doesn't; after then pairing b+l,
    b passes and a doesn't; that holds through repeated use; l alone never passes."""
    run = Run('learn2', gain, fps)
    log = lambda *a: verbose and print(*a)
    result = {'before_silent': not run.pulse('a') and not run.pulse('b')}
    for on, off in (('a', 'b'), ('b', 'a')):
        result[f'{on}_first_at'] = None
        for n in range(1, pairings + 1):
            run.pulse(on, 'l')
            ok = run.pulse(on) and not run.pulse(off)
            log(f'{on}+l pairing {n}: {on} passes and {off} does not: {ok}   {run.weights("ja", "jb")}')
            if ok and result[f'{on}_first_at'] is None:
                result[f'{on}_first_at'] = n
        result[f'{on}_reliable'] = all(run.pulse(on) and not run.pulse(off) for _ in range(5))
    result['l_alone_silent'] = not run.pulse('l')
    log(f"l alone silent {result['l_alone_silent']}   {run.weights('ja', 'jb')}")
    result['ok'] = all(result[k] for k in ('before_silent', 'a_reliable', 'b_reliable', 'l_alone_silent'))
    return result


if __name__ == '__main__':
    print(train_learn1(verbose=True))
    print(train_learn2(verbose=True))
