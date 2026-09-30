"""How a cell behaves. Everything else in the project is UI and support.

Every cell has:
  kind          EMPTY, NORMAL, GOOD or BAD (set by the designer; GOOD and BAD are teacher cells)
  energy        fuel, 0..1, refilled at SUPPLY/S
  flame         how strongly it is burning (0 = dark)
  weight        how readily it responds to light (1 = normal); what learning changes
  trace         when it last ignited: 1 when it does, fading with TRACE_TIME. Neighbours can read it; comparing
                two traces says only how far apart the two cells fired, never how long ago, so there is no cutoff
  delay         the slowest firing (longest delay after its cause) attention has met on its way back to this cell
  attention     counts down after the cell is credited as a cause of something the teacher valued: it is attended
                (passing attention on) for ATTENTION_TIME, then refractory for ATTENTION_REST, like a flame
  valence       +1 if the attention it last accepted came from a GOOD teacher (reward), -1 from a BAD one (punishment)
  suspicion     seconds it has spent next to an attended cell, having fired, without being accepted as a cause
  heat          its temperature above T_BASE: how readily it fires by chance while learning
and receives, each step, from the other cells:
  illumination  light from NORMAL cells' flames, which can ignite it
A teacher cell (GOOD or BAD) burns like a normal cell when struck, but gives no light and is never lit: striking it
pays attention to whatever just happened next to it, as a reward (GOOD) or a punishment (BAD). Outside inputs are
strikes (inputs and the teachers) and `teaching` (0 or 1), which switches learning on: chance firing, and all weight
and temperature changes.

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
     the one that ignited first in the CAUSE_WINDOW before it (among the cells two away, which light also reaches,
     if no adjacent one did), and offers the level trace^ATTENTION_TOL * earliest^(1 - ATTENTION_TOL) (a teacher,
     which nothing lit, offers its own trace: everything that fired beside it in the window before it counts). A cell
     listens to the attended neighbour that ignited soonest after it within CAUSE_WINDOW (one it could have caused);
     if its trace is at or below that neighbour's level, it becomes attended for ATTENTION_TIME, and then can't
     accept attention again for ATTENTION_REST (so one teacher strike sends one wave back). A cell that fired within
     CAUSE_WINDOW of an attended neighbour and stays next to it for BLAME_DELAY without being accepted is a leak: it
     fired beside a valued route without causing anything on it (backflow into a competing route, or a side leak).
  9. while teaching:
       a cell newly attended by a reward gains CREDIT weight and cools to T_BASE; by a punishment, loses PUNISH, but
       not below PUNISH_FLOOR
       (if WEAKEST > 0, punishment, and credit to a cell whose weight is at least RELIABLE, go only to a weakest link:
       a cell that fired more than WEAKEST times more slowly after its cause than any cell attention met on its way
       back from the teacher)
       a leak (beside a rewarded route) loses BLAME weight
       every ignition costs FIRE_COST weight (firing costs; being useful pays), and a cell that hasn't fired within
       TRACE_TIME gains RECOVER weight per second (quiet cells slowly become excitable again)
       a near miss (ready to fire, lit to NEAR_MISS of its threshold, but not igniting) warms at HEAT per second,
       unless attention is within light's reach (two cells), which cools it to T_BASE; heat fades with COOL_TIME
     weights stay within WEIGHT_MIN..WEIGHT_MAX, and T within T_BASE..T_MAX.

The step is written as plain loops over cells, each reading only its neighbours, and compiled by Numba (the first
call in a process takes a few seconds; the result is cached on disk). Inside the compiled functions only numbers and
arrays are allowed. To add a parameter, add it to DEFAULT_PARAMS and to the matching unpacking line in _step.
"""

from __future__ import annotations

import functools
import math

import numba
import numpy as np

EMPTY, NORMAL, GOOD, BAD = 0, 1, 2, 3
TEACHERS = (GOOD, BAD)

# The one parameter set shared by every bundled pattern (see README, "Designing Gates From First Principles").
# The order matters: _step unpacks them in this order.
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
    "CAUSE_WINDOW": 1.0,
    "ATTENTION_TIME": 1.0,
    "ATTENTION_REST": 8.0,
    "BLAME_DELAY": 0.2,
    "CREDIT": 0.2,
    "BLAME": 0.05,
    "PUNISH": 0.1,
    "PUNISH_FLOOR": 0.0,
    "WEAKEST": 0.0,
    "RELIABLE": 0.0,
    "FIRE_COST": 0.002,
    "RECOVER": 0.0,
    "WEIGHT_MIN": 0.0,
    "WEIGHT_MAX": 2.0,
    "CHANCE_RATE": 5.0,
    "T_BASE": 0.01,
    "T_MAX": 0.12,
    "HEAT": 0.02,
    "NEAR_MISS": 0.15,
    "COOL_TIME": 240.0,
}

# neighbours: the 8 adjacent cells, the 12 two away that light also reaches, and all cells within light's reach
RING1 = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)])
RING2 = np.array([(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2 and dx * dx + dy * dy <= 5])
LIGHT_REACH = np.array([(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if dx * dx + dy * dy <= 5])


@functools.lru_cache
def kernel(sigma: float) -> np.ndarray:
    """1-D Gaussian weights out to 4 sigma, summing to 1."""
    x = np.arange(-int(4 * sigma + 0.5), int(4 * sigma + 0.5) + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def step(cells: dict[str, np.ndarray], p: dict[str, float], dt: float, teaching: float = 0.0,
         rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Advance every cell by dt seconds. cells holds kind, energy, flame, weight, trace, delay, attention, valence,
    suspicion and heat; returns their new values plus this step's illumination. rng supplies chance firing while
    teaching."""
    learning = bool(teaching) and rng is not None
    noise = rng.random(cells["kind"].shape) if learning else np.ones(cells["kind"].shape)
    out = _step(cells["kind"], cells["energy"], cells["flame"], cells["weight"], cells["trace"], cells["delay"],
                cells["attention"], cells["valence"], cells["suspicion"], cells["heat"], noise, learning,
                kernel(p["COUPLING_DIST"]), tuple(float(p[name]) for name in DEFAULT_PARAMS), float(dt))
    return dict(zip(("energy", "flame", "weight", "trace", "delay", "attention", "valence", "suspicion", "heat",
                     "illumination"), out))


def strike(kind: int, energy: float, p: dict[str, float]) -> bool:
    """A manual strike obeys the same fuel rule as ignition by light. (The caller sets flame and trace, and a
    struck teacher becomes attended, with its valence: that is what a teacher is.)"""
    return kind != EMPTY and energy >= p["STRIKE_LEVEL"]


def stuck_on_risk(p: dict[str, float]) -> bool:
    """True if a flame can settle where burning balances refill, (SUPPLY/S)/FLAME_CONSUME, at or above MIN_FLAME, and burn forever."""
    return p["SUPPLY/S"] >= p["MIN_FLAME"] * p["FLAME_CONSUME"]


# --- the compiled per-cell rules ------------------------------------------------------------------------------------

@numba.njit(cache=True)
def _inside(x, y, w, h):
    return 0 <= x < w and 0 <= y < h


@numba.njit(cache=True)
def _light(flame, kind, k, gain):
    """Step 1: light from the other NORMAL cells' flames (a separable Gaussian blur, minus the cell's own flame)."""
    w, h = flame.shape
    r = (k.shape[0] - 1) // 2
    source = np.where(kind == NORMAL, flame, 0.0)
    rows = np.zeros((w, h))
    for x in range(w):
        for y in range(h):
            for i in range(-r, r + 1):
                if 0 <= y + i < h:
                    rows[x, y] += source[x, y + i] * k[i + r]
    light = np.zeros((w, h))
    own = k[r] * k[r]
    for x in range(w):
        for y in range(h):
            blur = 0.0
            for i in range(-r, r + 1):
                if 0 <= x + i < w:
                    blur += rows[x + i, y] * k[i + r]
            light[x, y] = max(0.0, gain * (blur - own * source[x, y]))
    return light


@numba.njit(cache=True)
def _attention(trace, attended, valence, slowest, receptive, teacher, tol, window):
    """Step 8: which cells accept attention this step (with the valence and slowest delay they hear, and their own
    delay after their cause), and which are leaking. window = exp(CAUSE_WINDOW / TRACE_TIME): the trace ratio of two
    cells that fired CAUSE_WINDOW apart. Each attended cell offers in one ring only: the adjacent one, or the next
    ring out if no adjacent cell fired before it."""
    w, h = trace.shape
    accept = np.zeros((w, h), np.bool_)
    heard_valence, heard_slowest, lag = np.zeros((w, h)), np.zeros((w, h)), np.zeros((w, h))
    leaking = np.zeros((w, h), np.bool_)
    if not attended.any():                      # most of the time: nothing to pass on, nothing to accept or leak
        return accept, heard_valence, heard_slowest, lag, leaking
    offered_before = np.zeros((w, h), np.bool_)
    level = np.zeros((w, h))
    for r in range(2):
        ring = RING1 if r == 0 else RING2
        # each cell: its earliest cause in this ring, how long after it the cell fired, and the level it offers
        for x in range(w):
            for y in range(h):
                t = trace[x, y]
                earliest = np.inf
                for i in range(ring.shape[0]):
                    nx, ny = x + ring[i, 0], y + ring[i, 1]
                    if _inside(nx, ny, w, h):
                        n = trace[nx, ny]
                        if n < t and n * window >= t and n < earliest:
                            earliest = n
                level[x, y] = 0.0
                if earliest < np.inf and not offered_before[x, y]:
                    lag[x, y] = t / earliest
                    if attended[x, y]:
                        level[x, y] = t if teacher[x, y] else t ** tol * earliest ** (1 - tol)
                if earliest < np.inf:
                    offered_before[x, y] = True
        # each cell listens to the attended neighbour that fired soonest after it, and accepts if it fired early enough
        for x in range(w):
            for y in range(h):
                t = trace[x, y]
                if accept[x, y] or not (t > 0 and receptive[x, y]):
                    continue
                soonest, heard, v, s = np.inf, 0.0, 0.0, 0.0
                for i in range(ring.shape[0]):
                    nx, ny = x + ring[i, 0], y + ring[i, 1]
                    if _inside(nx, ny, w, h):
                        n, n_level = trace[nx, ny], level[nx, ny]
                        if n_level > 0 and t < n <= t * window and n < soonest:
                            soonest, heard, v, s = n, n_level, valence[nx, ny], slowest[nx, ny]
                if t <= heard:
                    accept[x, y] = True
                    heard_valence[x, y], heard_slowest[x, y] = v, s
    # a leak fired within the window of a rewarded attended neighbour and wasn't accepted
    for x in range(w):
        for y in range(h):
            t = trace[x, y]
            if t > 0 and receptive[x, y] and not accept[x, y]:
                for i in range(RING1.shape[0]):
                    nx, ny = x + RING1[i, 0], y + RING1[i, 1]
                    if _inside(nx, ny, w, h) and attended[nx, ny] and valence[nx, ny] > 0:
                        n = trace[nx, ny]
                        if n <= t * window and n * window >= t:
                            leaking[x, y] = True
    return accept, heard_valence, heard_slowest, lag, leaking


@numba.njit(cache=True)
def _step(kind, energy, flame, weight, trace, delay, attention, valence, suspicion, heat, noise, learning, k, P, dt):
    (COUPLING_DIST, COUPLING_GAIN, FLAME_CONSUME, FLAME_INERTIA, MIN_FLAME, MIN_STRIKE, STRIKE_LEVEL, SUPPLY, TRACE_TIME,
     ATTENTION_TOL, CAUSE_WINDOW, ATTENTION_TIME, ATTENTION_REST, BLAME_DELAY, CREDIT, BLAME, PUNISH, PUNISH_FLOOR, WEAKEST,
     RELIABLE, FIRE_COST, RECOVER, WEIGHT_MIN, WEIGHT_MAX, CHANCE_RATE, T_BASE, T_MAX, HEAT, NEAR_MISS, COOL_TIME) = P
    w, h = kind.shape
    energy, flame, weight, heat = energy.copy(), flame.copy(), weight.copy(), heat.copy()

    illumination = _light(flame, kind, k, COUPLING_GAIN)                                              # 1

    ready, ignite, ignited = np.zeros((w, h), np.bool_), np.zeros((w, h), np.bool_), np.zeros((w, h), np.bool_)
    for x in range(w):
        for y in range(h):
            was_burning = flame[x, y] > 0
            ready[x, y] = kind[x, y] == NORMAL and flame[x, y] == 0 and energy[x, y] >= STRIKE_LEVEL
            drive = weight[x, y] * illumination[x, y] / MIN_STRIKE
            ignite[x, y] = ready[x, y] and drive >= 1                                                  # 2
            if learning and ready[x, y] and not ignite[x, y]:
                rate = CHANCE_RATE * math.exp(min(drive - 1, 0.0) / (T_BASE + heat[x, y]))
                ignite[x, y] = noise[x, y] < -math.expm1(-rate * dt)
            f, e = flame[x, y], energy[x, y]
            if ignite[x, y]:
                f = STRIKE_LEVEL
            if f < MIN_FLAME:                                                                            # 3
                f = 0.0
            if f > 0:                                                                                    # 4
                f = e + (f - e) * math.exp(-dt / FLAME_INERTIA)
            e -= f * FLAME_CONSUME * dt                                                                  # 5
            if e <= 0:
                f, e = 0.0, 0.0
            if kind[x, y] != EMPTY:                                                                      # 6
                e = min(1.0, e + SUPPLY * dt)
            flame[x, y], energy[x, y] = f, e
            ignited[x, y] = ignite[x, y] and not was_burning

    fade = math.exp(-dt / TRACE_TIME)                                                                    # 7
    trace = np.where(ignited, 1.0, trace * fade)

    attended = attention > ATTENTION_REST                                                                # 8
    teacher = (kind == GOOD) | (kind == BAD)
    accept, heard_valence, heard_slowest, lag, leaking = _attention(
        trace, attended, valence, delay, attention <= 0, teacher, ATTENTION_TOL, math.exp(CAUSE_WINDOW / TRACE_TIME))
    valence, delay, attention, suspicion = valence.copy(), delay.copy(), attention.copy(), suspicion.copy()
    rewarded, punished, leak = np.zeros((w, h), np.bool_), np.zeros((w, h), np.bool_), np.zeros((w, h), np.bool_)
    for x in range(w):
        for y in range(h):
            if accept[x, y]:
                valence[x, y] = heard_valence[x, y]
                delay[x, y] = max(lag[x, y], heard_slowest[x, y])     # the slowest firing met so far on the way back
                attention[x, y] = ATTENTION_TIME + ATTENTION_REST
                weakest = WEAKEST == 0 or lag[x, y] > heard_slowest[x, y] * WEAKEST
                rewarded[x, y] = heard_valence[x, y] > 0 and (weakest or weight[x, y] < RELIABLE)
                punished[x, y] = heard_valence[x, y] < 0 and weakest
            else:
                attention[x, y] = max(attention[x, y] - dt, 0.0)
            before = suspicion[x, y]
            suspicion[x, y] = before + dt if leaking[x, y] else 0.0
            leak[x, y] = leaking[x, y] and before < BLAME_DELAY <= suspicion[x, y]

    if learning:                                                                                         # 9
        near_attention = np.zeros((w, h), np.bool_)
        for x in range(w):
            for y in range(h):
                if attention[x, y] > ATTENTION_REST and valence[x, y] > 0:
                    for i in range(LIGHT_REACH.shape[0]):
                        nx, ny = x + LIGHT_REACH[i, 0], y + LIGHT_REACH[i, 1]
                        if _inside(nx, ny, w, h):
                            near_attention[nx, ny] = True
        cool = math.exp(-dt / COOL_TIME)
        fade_out = math.exp(-1.0)                           # a trace this faded: not fired within TRACE_TIME
        for x in range(w):
            for y in range(h):
                wv = weight[x, y] + CREDIT * rewarded[x, y] - BLAME * leak[x, y] - FIRE_COST * ignited[x, y]
                if trace[x, y] < fade_out:                  # quiet for a while: slowly become more excitable again
                    wv += RECOVER * dt
                if punished[x, y]:                          # punishment never takes a cell below PUNISH_FLOOR
                    wv = max(wv - PUNISH, min(wv, PUNISH_FLOOR))
                if kind[x, y] == NORMAL:
                    wv = min(max(wv, WEIGHT_MIN), WEIGHT_MAX)
                weight[x, y] = wv
                near_miss = ready[x, y] and not ignite[x, y] and wv * illumination[x, y] >= NEAR_MISS * MIN_STRIKE
                hot = heat[x, y] * cool + HEAT * dt * (near_miss and not near_attention[x, y])
                heat[x, y] = 0.0 if (near_attention[x, y] or rewarded[x, y]) else min(hot, T_MAX - T_BASE)
    return energy, flame, weight, trace, delay, attention, valence, suspicion, heat, illumination
