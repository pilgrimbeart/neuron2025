"""How a cell behaves. Everything else in the project is UI and support.

Every cell holds three numbers:
  energy        fuel, 0..1, refilled at SUPPLY/S while the cell is enabled
  flame         how strongly it is burning (0 = dark)
  illumination  light arriving from nearby flames

Each time step (dt seconds):
  1. illumination = COUPLING_GAIN * Gaussian blur of flame, with radius COUPLING_DIST
  2. a dark enabled cell with at least STRIKE_LEVEL energy ignites (flame = STRIKE_LEVEL)
     if its illumination reaches MIN_STRIKE
  3. a flame weaker than MIN_FLAME goes out
  4. a burning flame relaxes toward the cell's energy with time constant FLAME_INERTIA
  5. burning uses energy at FLAME_CONSUME * flame; a flame whose fuel runs out goes out
  6. enabled cells refill energy at SUPPLY/S, up to 1
"""

from __future__ import annotations

import math

import numpy as np
import scipy.ndimage


# The one parameter set shared by every bundled pattern (see README, "Designing Gates From First Principles").
DEFAULT_PARAMS = {
    "COUPLING_DIST": 0.85,
    "COUPLING_GAIN": 6.0,
    "FLAME_CONSUME": 3.0,
    "FLAME_INERTIA": 1.0,
    "MIN_FLAME": 0.06,
    "MIN_STRIKE": 0.2,
    "STRIKE_LEVEL": 0.12,
    "SUPPLY/S": 0.144,
}


def step(enabled: np.ndarray, energy: np.ndarray, flame: np.ndarray, p: dict[str, float], dt: float):
    """Advance every cell by dt seconds. Returns new (energy, flame, illumination) arrays."""
    energy, flame = energy.copy(), flame.copy()

    illumination = p["COUPLING_GAIN"] * scipy.ndimage.gaussian_filter(flame, sigma=p["COUPLING_DIST"], mode="constant")

    ignite = enabled & (flame == 0) & (energy >= p["STRIKE_LEVEL"]) & (illumination >= p["MIN_STRIKE"])
    flame[ignite] = p["STRIKE_LEVEL"]

    flame[flame < p["MIN_FLAME"]] = 0

    burning = flame > 0
    decay = math.exp(-dt / p["FLAME_INERTIA"])
    flame[burning] = energy[burning] + (flame[burning] - energy[burning]) * decay

    energy -= flame * p["FLAME_CONSUME"] * dt
    exhausted = energy <= 0
    flame[exhausted] = 0
    energy[exhausted] = 0

    energy[enabled] = np.minimum(1, energy[enabled] + p["SUPPLY/S"] * dt)
    return energy, flame, illumination


def can_ignite(enabled: bool, energy: float, p: dict[str, float]) -> bool:
    """A manual strike obeys the same fuel rule as ignition by light."""
    return bool(enabled) and energy >= p["STRIKE_LEVEL"]


def stuck_on_risk(p: dict[str, float]) -> bool:
    """True if a flame can settle where burning balances refill, (SUPPLY/S)/FLAME_CONSUME, at or above MIN_FLAME, and burn forever."""
    return p["SUPPLY/S"] >= p["MIN_FLAME"] * p["FLAME_CONSUME"]
