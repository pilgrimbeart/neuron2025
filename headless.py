"""Run the simulator without pygame, for scripted experiments and regression checks."""

from __future__ import annotations

import numpy as np

from model import SimulationConfig, State
from persistence import load_snapshot


def load_pattern(name: str) -> tuple[State, list[dict], SimulationConfig]:
    state, probes, vars_dict = load_snapshot(name)
    config = SimulationConfig()
    config.update_from_dict(vars_dict)
    return state, probes, config


def probe_by_label(probes: list[dict], label: str) -> dict | None:
    for probe in probes:
        if probe.get("label") == label:
            return probe
    return None


def grid_summary(state: State) -> tuple[int, float]:
    """(number of currently-lit cells, total flame)."""
    return int((state.flame_array > 0).sum()), float(state.flame_array.sum())


def simulate(state: State, config: SimulationConfig, strikes: dict[float, list[tuple[int, int]]], seconds: float, dt: float) -> np.ndarray:
    """Strike cells at the given times (seconds), run, and return how many times each cell ignited.

    A struck cell counts as igniting. state is advanced in place.
    """
    schedule: dict[int, list[tuple[int, int]]] = {}
    for t, cells in strikes.items():
        schedule.setdefault(round(t / dt), []).extend(cells)
    ignitions = np.zeros(state.grid_size, dtype=int)
    for step in range(round(seconds / dt)):
        for xy in schedule.get(step, []):
            if state.strike(xy, config):
                ignitions[xy] += 1
        before = state.flame_array > 0
        state.update(dt, config)
        ignitions += (state.flame_array > 0) & ~before
    return ignitions
