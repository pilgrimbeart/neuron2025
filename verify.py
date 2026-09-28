"""Regression suite: every bundled pattern, one shared parameter set, behaviour checked with margins.

    python verify.py              check everything
    python verify.py xor cross    check just those

For each pattern, the saved file must use physics.DEFAULT_PARAMS and match its layout in gates.py, and
every case below must hold at the nominal parameters, with COUPLING_GAIN changed by -20% and +20%,
and at 30 and 100 frames per second. Unless a pattern is expected to keep running, every pulse must
also burn each cell at most once and die out completely.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np

import gates
import physics
from headless import load_pattern, simulate
from model import SimulationConfig

# (strikes at t=0, expected ignition counts at probes). Struck probes are not checked.
CASES = {
    'line': [(['in'], {'out': 1})],
    'oneway': [(['a'], {'b': 1}), (['b'], {'a': 0})],
    'and': [(['a'], {'out': 0, 'b': 0}), (['b'], {'out': 0, 'a': 0}), (['a', 'b'], {'edge': 1}), (['edge'], {'a': 0, 'b': 0})],
    'or': [(['a'], {'edge': 1, 'b': 0}), (['b'], {'edge': 1, 'a': 0})],
    'xor': [(['a'], {'edge': 1, 'b': 0}), (['b'], {'edge': 1, 'a': 0}), (['a', 'b'], {'edge': 0})],
    'cross': [(['a'], {'a_out': 1, 'b_out': 0, 'b': 0}), (['b'], {'b_out': 1, 'a_out': 0, 'a': 0}), (['a', 'b'], {'a_out': 1, 'b_out': 1})],
    'inhibit': [(['in'], {'out': 1, 'inh': 0}), (['inh'], {'out': 0, 'in': 0}), (['in', 'inh'], {'out': 0})],
    'osc': [(['start'], {'out': (3, None)})],     # (min, max): keeps pulsing
}
KEEPS_RUNNING = {'osc'}
SECONDS = {'cross': 40, 'osc': 60}
VARIANTS = [(1.0, 50), (0.8, 50), (1.2, 50), (1.0, 30), (1.0, 100)]    # (gain factor, frames per second)


def check_file(name: str) -> list[str]:
    problems = []
    state, probes, config = load_pattern(name)
    if config.vars != physics.DEFAULT_PARAMS:
        problems.append(f'parameters differ from physics.DEFAULT_PARAMS: {config.vars}')
    expected, expected_probes = gates.build_state(name)
    if state.grid_size != expected.grid_size or not np.array_equal(state.kind_array, expected.kind_array):
        problems.append('layout differs from gates.py (run: python gates.py ' + name + ')')
    if {p.get('label'): tuple(p['xy']) for p in probes} != {p['label']: p['xy'] for p in expected_probes}:
        problems.append('probes differ from gates.py')
    return problems


def run_case(job) -> tuple[str, str | None]:
    name, strikes, expect, gain, fps = job
    state, probes, _ = load_pattern(name)
    state.reset_energy_and_flame()
    config = SimulationConfig(dict(physics.DEFAULT_PARAMS))
    config.set('COUPLING_GAIN', config.get('COUPLING_GAIN') * gain)
    at = {p['label']: tuple(p['xy']) for p in probes}
    ignitions = simulate(state, config, {0: [at[label] for label in strikes]}, SECONDS.get(name, 30), 1 / fps)

    where = f"strike {'+'.join(strikes)}, gain x{gain:g}, {fps} fps"
    for label, want in expect.items():
        got = int(ignitions[at[label]])
        low, high = want if isinstance(want, tuple) else (want, want)
        if got < low or (high is not None and got > high):
            return name, f'{where}: {label} fired {got} times, expected {want}'
    lit = int((state.flame_array > 0).sum())
    if name in KEEPS_RUNNING:
        if lit == 0:
            return name, f'{where}: stopped running'
    else:
        if ignitions.max() > 1:
            return name, f'{where}: a cell ignited {int(ignitions.max())} times'
        if lit:
            return name, f'{where}: {lit} cells still burning at the end'
    return name, None


def main(names: list[str]) -> int:
    failures = {name: check_file(name) for name in names}
    jobs = [(name, strikes, expect, gain, fps) for name in names for strikes, expect in CASES[name] for gain, fps in VARIANTS]
    with Pool() as pool:
        for name, problem in pool.imap_unordered(run_case, jobs):
            if problem:
                failures[name].append(problem)
    for name in names:
        runs = len(CASES[name]) * len(VARIANTS)
        print(f"{'PASS' if not failures[name] else 'FAIL'}  {name:8} ({runs} runs)")
        for problem in failures[name]:
            print(f'      {problem}')
    return 1 if any(failures.values()) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or list(gates.PATTERNS)))
