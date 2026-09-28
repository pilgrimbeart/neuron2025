"""How a cell behaves. Everything else in the project is UI and support.

Every cell has:
  kind          EMPTY, NORMAL, or TRANSDUCER (set by the designer)
  energy        fuel, 0..1, refilled at SUPPLY/S
  flame         how strongly it is burning (0 = dark)
  weight        how readily it responds to light (1 = normal); the slow value learning changes
  light_trace   fading memory of the brightest recent illumination
  teach_trace   fading memory of the brightest recent teaching signal
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
  7. each trace follows its signal up at once, and otherwise fades with time constant TRACE_TIME
  8. learning, with teaching trace T and light trace L, kept within WEIGHT_MIN..WEIGHT_MAX:
       every cell strengthens at LEARN_RATE * T * L   (credit for light received, whether it then fired or not)
       a cell at rest (dark, full energy) weakens at UNLEARN_RATE * T * T   (teaching signal in the dark)
     Because it uses traces, light and teaching signal count as together if within a couple of seconds.
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
    "TRACE_TIME": 2.0,
}


@functools.lru_cache
def kernel(sigma: float) -> np.ndarray:
    """1-D Gaussian weights out to 4 sigma, summing to 1."""
    x = np.arange(-int(4 * sigma + 0.5), int(4 * sigma + 0.5) + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def spread(sources: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Light each cell receives from the other cells' source values: a Gaussian blur over the last two axes,
    minus the cell's own contribution."""
    k = kernel(p["COUPLING_DIST"])
    blur = scipy.ndimage.correlate1d(sources, k, axis=-1, mode="constant")
    blur = scipy.ndimage.correlate1d(blur, k, axis=-2, mode="constant")
    own = k[len(k) // 2] ** 2
    return np.maximum(0.0, p["COUPLING_GAIN"] * (blur - own * sources))


def step(cells: dict[str, np.ndarray], p: dict[str, float], dt: float) -> dict[str, np.ndarray]:
    """Advance every cell by dt seconds. cells holds kind, energy, flame, weight, light_trace and teach_trace;
    returns their new values plus this step's illumination and modulator."""
    kind, weight = cells["kind"], cells["weight"]
    energy, flame = cells["energy"].copy(), cells["flame"].copy()
    present = kind != EMPTY

    illumination, modulator = spread(np.stack([np.where(kind == NORMAL, flame, 0.0), np.where(kind == TRANSDUCER, flame, 0.0)]), p)

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

    fade = math.exp(-dt / p["TRACE_TIME"])
    light_trace = np.maximum(illumination, cells["light_trace"] * fade)
    teach_trace = np.maximum(modulator, cells["teach_trace"] * fade)

    at_rest = present & (flame == 0) & (energy >= 1.0)
    strengthen = p["LEARN_RATE"] * teach_trace * light_trace * present
    weaken = p["UNLEARN_RATE"] * teach_trace * teach_trace * at_rest
    weight = np.clip(weight + (strengthen - weaken) * dt, p["WEIGHT_MIN"], p["WEIGHT_MAX"])
    return {"energy": energy, "flame": flame, "weight": weight, "light_trace": light_trace, "teach_trace": teach_trace,
            "illumination": illumination, "modulator": modulator}


def can_ignite(kind: int, energy: float, p: dict[str, float]) -> bool:
    """A manual strike obeys the same fuel rule as ignition by light."""
    return kind != EMPTY and energy >= p["STRIKE_LEVEL"]


def stuck_on_risk(p: dict[str, float]) -> bool:
    """True if a flame can settle where burning balances refill, (SUPPLY/S)/FLAME_CONSUME, at or above MIN_FLAME, and burn forever."""
    return p["SUPPLY/S"] >= p["MIN_FLAME"] * p["FLAME_CONSUME"]
