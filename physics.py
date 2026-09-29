"""How a cell behaves. Everything else in the project is UI and support.

Every cell has:
  kind          EMPTY, NORMAL, or TEACHER (set by the designer)
  energy        fuel, 0..1, refilled at SUPPLY/S
  flame         how strongly it is burning (0 = dark)
  weight        how readily it responds to light (1 = normal); what learning changes
  trace         how recently it ignited: 1 when it does, fading with TRACE_TIME. Neighbours can read it.
  attention     counts down after the cell is credited as a cause of something the teacher valued: it is attended
                (passing attention on) for ATTENTION_TIME, then refractory for ATTENTION_REST, like a flame
  suspicion     seconds it has spent next to an attended cell, having fired, without being accepted as a cause
  heat          its temperature above T_BASE: how readily it fires by chance while learning
and receives, each step, from the other cells:
  illumination  light from NORMAL cells' flames, which can ignite it
A TEACHER cell burns like a normal cell when struck, but gives no light and is never lit: striking it pays attention
to whatever just happened next to it. Outside inputs are strikes (inputs and the teacher) and `teaching` (0 or 1),
which switches learning on: chance firing, and all weight and temperature changes.

Each time step (dt seconds):
  1. illumination = COUPLING_GAIN * Gaussian blur, with radius COUPLING_DIST, of the other NORMAL cells' flames
  2. a dark NORMAL cell with at least STRIKE_LEVEL energy ignites (flame = STRIKE_LEVEL)
     if weight * illumination reaches MIN_STRIKE; while teaching it may also ignite by chance, at
     CHANCE_RATE * exp((x - 1) / T) per second, where x = weight * illumination / MIN_STRIKE and T = T_BASE + heat
  3. a flame weaker than MIN_FLAME goes out
  4. a burning flame relaxes toward the cell's energy with time constant FLAME_INERTIA
  5. burning uses energy at FLAME_CONSUME * flame; a flame whose fuel runs out goes out
  6. non-empty cells refill energy at SUPPLY/S, up to 1
  7. trace: 1 for a cell that ignited, otherwise fading
  8. attention passes one hop back, from effects to causes. An attended cell looks at its neighbours' traces, finds
     the one that ignited first before it (among the cells two away, which light also reaches, if no adjacent one
     did), and offers the level trace^ATTENTION_TOL * earliest^(1 - ATTENTION_TOL) (a teacher, which nothing lit,
     offers its own trace: everything that fired beside it before it counts). A cell that fired listens to the
     attended neighbour that ignited soonest after it (the one it could have caused); if its own trace is at or below
     that neighbour's level, it becomes attended for ATTENTION_TIME, and then can't accept attention again for
     ATTENTION_REST (so one teacher strike sends one wave back). A cell that fired and stays next to an attended
     cell for BLAME_DELAY without being accepted is a leak: it fired beside a valued route without causing anything
     on it (backflow into a competing route, or a side leak).
  9. while teaching:
       a newly attended cell gains CREDIT weight and cools to T_BASE; a leak loses BLAME weight
       every ignition costs FIRE_COST weight (firing costs; being useful pays)
       a near miss (lit to NEAR_MISS of its threshold, dark, not fired recently) warms at HEAT per second,
       unless attention is within light's reach (two cells), which cools it to T_BASE; heat fades with COOL_TIME
     weights stay within WEIGHT_MIN..WEIGHT_MAX, and T within T_BASE..T_MAX.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import scipy.ndimage

EMPTY, NORMAL, TEACHER = 0, 1, 2

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
    "TRACE_TIME": 3.0,
    "ATTENTION_TOL": 0.3,
    "ATTENTION_TIME": 1.0,
    "ATTENTION_REST": 8.0,
    "BLAME_DELAY": 0.2,
    "CREDIT": 0.2,
    "BLAME": 0.05,
    "FIRE_COST": 0.002,
    "WEIGHT_MIN": 0.0,
    "WEIGHT_MAX": 2.0,
    "CHANCE_RATE": 5.0,
    "T_BASE": 0.01,
    "T_MAX": 0.12,
    "HEAT": 0.02,
    "NEAR_MISS": 0.15,
    "COOL_TIME": 240.0,
}

FIRED = 0.05     # a trace above this means the cell fired recently enough to count

RING1 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
RING2 = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2 and dx * dx + dy * dy <= 5]
LIGHT_REACH = np.array([[dx * dx + dy * dy <= 5 for dy in range(-2, 3)] for dx in range(-2, 3)])


@functools.lru_cache
def kernel(sigma: float) -> np.ndarray:
    """1-D Gaussian weights out to 4 sigma, summing to 1."""
    x = np.arange(-int(4 * sigma + 0.5), int(4 * sigma + 0.5) + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def spread(sources: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Light each cell receives from the other cells' flames: a Gaussian blur minus the cell's own contribution."""
    k = kernel(p["COUPLING_DIST"])
    blur = scipy.ndimage.correlate1d(sources, k, axis=-1, mode="constant")
    blur = scipy.ndimage.correlate1d(blur, k, axis=-2, mode="constant")
    own = k[len(k) // 2] ** 2
    return np.maximum(0.0, p["COUPLING_GAIN"] * (blur - own * sources))


def neighbour(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Each cell's view of its neighbour at offset (dx, dy): b[x, y] = a[x + dx, y + dy], 0 beyond the edge."""
    b = np.zeros_like(a)
    w, h = a.shape
    b[max(0, -dx):min(w, w - dx), max(0, -dy):min(h, h - dy)] = a[max(0, dx):min(w, w + dx), max(0, dy):min(h, h + dy)]
    return b


def pass_attention(trace: np.ndarray, attended: np.ndarray, receptive: np.ndarray, teacher: np.ndarray,
                   tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Step 8: which cells accept attention this step, and which are leaking (fired, next to an attended cell, not
    accepted). Only receptive cells (neither attended nor refractory) can accept or leak."""
    fired = (trace > FIRED) & receptive
    accept = np.zeros(trace.shape, dtype=bool)
    offered_before = np.zeros(trace.shape, dtype=bool)
    for ring in (RING1, RING2):
        earliest = np.full(trace.shape, np.inf)
        for dx, dy in ring:
            n = neighbour(trace, dx, dy)
            earliest = np.where((n > FIRED) & (n < trace), np.minimum(earliest, n), earliest)
        offers = attended & np.isfinite(earliest) & ~offered_before      # each attended cell offers in one ring only
        offered_before |= np.isfinite(earliest)
        level = np.where(offers, trace ** tol * np.where(offers, earliest, 1.0) ** (1 - tol), 0.0)
        level = np.where(offers & teacher, trace, level)     # a teacher attends to everything that just fired beside it
        soonest_after, heard = np.full(trace.shape, np.inf), np.zeros(trace.shape)
        for dx, dy in ring:
            n, n_level = neighbour(trace, dx, dy), neighbour(level, dx, dy)
            closer = (n_level > 0) & (n > trace) & (n < soonest_after)
            soonest_after = np.where(closer, n, soonest_after)
            heard = np.where(closer, n_level, heard)
        accept |= fired & (trace <= heard)
    next_to_attended = np.zeros(trace.shape, dtype=bool)
    for dx, dy in RING1:
        next_to_attended |= neighbour(attended, dx, dy)
    return accept, fired & ~accept & next_to_attended


def step(cells: dict[str, np.ndarray], p: dict[str, float], dt: float, teaching: float = 0.0,
         rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Advance every cell by dt seconds. cells holds kind, energy, flame, weight, trace, attention, suspicion and
    heat; returns their new values plus this step's illumination. rng supplies chance firing while teaching."""
    kind = cells["kind"]
    energy, flame, weight = cells["energy"].copy(), cells["flame"].copy(), cells["weight"].copy()
    heat = cells["heat"].copy()
    present, normal = kind != EMPTY, kind == NORMAL
    was_burning = flame > 0

    illumination = spread(np.where(normal, flame, 0.0), p)

    ready = normal & (flame == 0) & (energy >= p["STRIKE_LEVEL"])
    ignite = ready & (weight * illumination >= p["MIN_STRIKE"])
    if teaching and rng is not None:
        x = weight * illumination / p["MIN_STRIKE"]
        rate = p["CHANCE_RATE"] * np.exp(np.minimum(x - 1, 0) / (p["T_BASE"] + heat))
        ignite |= ready & (rng.random(kind.shape) < -np.expm1(-rate * dt))
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

    ignited = ignite & ~was_burning
    trace = np.where(ignited, 1.0, cells["trace"] * math.exp(-dt / p["TRACE_TIME"]))

    attended = cells["attention"] > p["ATTENTION_REST"]
    accept, leaking = pass_attention(trace, attended, cells["attention"] <= 0, kind == TEACHER, p["ATTENTION_TOL"])
    attention = np.where(accept, p["ATTENTION_TIME"] + p["ATTENTION_REST"], np.maximum(cells["attention"] - dt, 0.0))
    suspicion = np.where(leaking, cells["suspicion"] + dt, 0.0)
    leak = leaking & (cells["suspicion"] < p["BLAME_DELAY"]) & (suspicion >= p["BLAME_DELAY"])

    if teaching:
        weight += p["CREDIT"] * accept - p["BLAME"] * leak - p["FIRE_COST"] * ignited
        weight = np.where(normal, np.clip(weight, p["WEIGHT_MIN"], p["WEIGHT_MAX"]), weight)
        attention_near = scipy.ndimage.binary_dilation(attention > p["ATTENTION_REST"], LIGHT_REACH)
        near_miss = normal & (flame == 0) & (trace < FIRED) & (weight * illumination >= p["NEAR_MISS"] * p["MIN_STRIKE"])
        heat = heat * math.exp(-dt / p["COOL_TIME"]) + p["HEAT"] * dt * (near_miss & ~attention_near)
        heat = np.where(attention_near | accept, 0.0, np.minimum(heat, p["T_MAX"] - p["T_BASE"]))
    return {"energy": energy, "flame": flame, "weight": weight, "trace": trace, "attention": attention,
            "suspicion": suspicion, "heat": heat, "illumination": illumination}


def strike(kind: int, energy: float, p: dict[str, float]) -> bool:
    """A manual strike obeys the same fuel rule as ignition by light. (The caller sets flame and trace, and a
    struck TEACHER becomes attended: that is what a teacher is.)"""
    return kind != EMPTY and energy >= p["STRIKE_LEVEL"]


def stuck_on_risk(p: dict[str, float]) -> bool:
    """True if a flame can settle where burning balances refill, (SUPPLY/S)/FLAME_CONSUME, at or above MIN_FLAME, and burn forever."""
    return p["SUPPLY/S"] >= p["MIN_FLAME"] * p["FLAME_CONSUME"]
