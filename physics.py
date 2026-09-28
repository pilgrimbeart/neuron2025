"""How a cell behaves. Everything else in the project is UI and support.

Every cell has:
  kind          EMPTY, NORMAL, or TRANSDUCER (set by the designer)
  energy        fuel, 0..1, refilled at SUPPLY/S
  flame         how strongly it is burning (0 = dark)
  weight        how readily it responds to light (1 = normal); the slow value learning will change
and receives two kinds of light, recomputed each step from other cells' flames (never its own):
  illumination  light from NORMAL cells, which can ignite it
  modulator     light from TRANSDUCER cells, which never ignites anything; a teaching signal

Each time step (dt seconds):
  1. illumination and modulator = COUPLING_GAIN * Gaussian blur, with radius COUPLING_DIST,
     of the other NORMAL and TRANSDUCER cells' flames respectively
  2. a dark cell with at least STRIKE_LEVEL energy ignites (flame = STRIKE_LEVEL)
     if weight * illumination reaches MIN_STRIKE
  3. a flame weaker than MIN_FLAME goes out
  4. a burning flame relaxes toward the cell's energy with time constant FLAME_INERTIA
  5. burning uses energy at FLAME_CONSUME * flame; a flame whose fuel runs out goes out
  6. non-empty cells refill energy at SUPPLY/S, up to 1
  7. learning: a ready cell (dark, with at least STRIKE_LEVEL energy) under teaching signal M changes its
     weight at M * (LEARN_RATE * illumination - UNLEARN_RATE * M), kept within WEIGHT_MIN..WEIGHT_MAX.
     Light brighter than the teaching signal strengthens; teaching signal in the dark weakens.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import scipy.ndimage

EMPTY, NORMAL, TRANSDUCER = 0, 1, 2

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
    "LEARN_RATE": 96.0,
    "UNLEARN_RATE": 144.0,
    "WEIGHT_MIN": 0.5,
    "WEIGHT_MAX": 2.0,
}


@functools.lru_cache
def self_weight(sigma: float) -> float:
    """The share of a cell's own flame that the Gaussian blur gives back to it."""
    impulse = np.zeros((9, 9))
    impulse[4, 4] = 1.0
    return float(scipy.ndimage.gaussian_filter(impulse, sigma=sigma, mode="constant")[4, 4])


def spread(source: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Light each cell receives from the other cells' source values."""
    blur = scipy.ndimage.gaussian_filter(source, sigma=p["COUPLING_DIST"], mode="constant")
    return np.maximum(0.0, p["COUPLING_GAIN"] * (blur - self_weight(p["COUPLING_DIST"]) * source))


def step(kind: np.ndarray, energy: np.ndarray, flame: np.ndarray, weight: np.ndarray, p: dict[str, float], dt: float):
    """Advance every cell by dt seconds. Returns new (energy, flame, weight, illumination, modulator) arrays."""
    energy, flame = energy.copy(), flame.copy()
    present = kind != EMPTY

    illumination = spread(np.where(kind == NORMAL, flame, 0.0), p)
    modulator = spread(np.where(kind == TRANSDUCER, flame, 0.0), p)

    ignite = present & (flame == 0) & (energy >= p["STRIKE_LEVEL"]) & (weight * illumination >= p["MIN_STRIKE"])
    flame[ignite] = p["STRIKE_LEVEL"]

    flame[flame < p["MIN_FLAME"]] = 0

    burning = flame > 0
    decay = math.exp(-dt / p["FLAME_INERTIA"])
    flame[burning] = energy[burning] + (flame[burning] - energy[burning]) * decay

    energy -= flame * p["FLAME_CONSUME"] * dt
    exhausted = energy <= 0
    flame[exhausted] = 0
    energy[exhausted] = 0

    energy[present] = np.minimum(1, energy[present] + p["SUPPLY/S"] * dt)

    ready = present & (flame == 0) & (energy >= p["STRIKE_LEVEL"])
    change = modulator * (p["LEARN_RATE"] * illumination - p["UNLEARN_RATE"] * modulator) * dt
    weight = np.where(ready, np.clip(weight + change, p["WEIGHT_MIN"], p["WEIGHT_MAX"]), weight)
    return energy, flame, weight, illumination, modulator


def can_ignite(kind: int, energy: float, p: dict[str, float]) -> bool:
    """A manual strike obeys the same fuel rule as ignition by light."""
    return kind != EMPTY and energy >= p["STRIKE_LEVEL"]


def stuck_on_risk(p: dict[str, float]) -> bool:
    """True if a flame can settle where burning balances refill, (SUPPLY/S)/FLAME_CONSUME, at or above MIN_FLAME, and burn forever."""
    return p["SUPPLY/S"] >= p["MIN_FLAME"] * p["FLAME_CONSUME"]
