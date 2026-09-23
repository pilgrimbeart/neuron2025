from __future__ import annotations

"""Run the simulator without pygame, for scripted analysis and training.

model.py and persistence.py have no pygame dependency, so a pattern can be
loaded and stepped forward directly. This module formalizes that pattern:
load a snapshot, optionally inject strikes at scheduled steps, step the
simulation, and record a trace at each probe -- the same (energy, flame,
illumination) triple the on-screen chart plots, just headless.
"""

from typing import Callable

from model import SimulationConfig, State
from persistence import load_snapshot


Trace = list[tuple[float, float, float]]


def load_config(vars_dict: dict[str, float]) -> SimulationConfig:
    config = SimulationConfig()
    config.update_from_dict(vars_dict)
    return config


def probe_by_label(probes: list[dict], label: str) -> dict | None:
    for probe in probes:
        if probe.get("label") == label:
            return probe
    return None


def probe_key(probe: dict, index: int) -> str:
    return probe.get("label") or str(index)


def run_trace(
    state: State,
    config: SimulationConfig,
    probes: list[dict],
    steps: int,
    dt: float = 1 / 50.0,
    on_step: Callable[[int, State], None] | None = None,
) -> dict[str, Trace]:
    """Step the simulation, recording (energy, flame, illumination) per probe.

    on_step(step, state), if given, runs before each update -- e.g. to strike
    a cell at a scheduled step.
    """
    traces: dict[str, Trace] = {probe_key(probe, index): [] for index, probe in enumerate(probes)}
    for step in range(steps):
        if on_step is not None:
            on_step(step, state)
        state.update(dt, config)
        for index, probe in enumerate(probes):
            xy = probe["xy"]
            traces[probe_key(probe, index)].append(
                (float(state.energy_array[xy]), float(state.flame_array[xy]), float(state.illumination_array[xy]))
            )
    return traces


def grid_summary(state: State) -> tuple[int, float]:
    """(number of currently-lit cells, total flame) -- a probe-free sanity check."""
    lit = int((state.flame_array > 0).sum())
    total_flame = float(state.flame_array.sum())
    return lit, total_flame


def load_pattern(name: str) -> tuple[State, list[dict], SimulationConfig]:
    state, probes, vars_dict = load_snapshot(name)
    return state, probes, load_config(vars_dict)
