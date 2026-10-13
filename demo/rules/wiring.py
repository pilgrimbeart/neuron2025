"""Wiring, from averages only: paths grow from R and B to A (channel 0) and to V (channel 1), and pulses run down them.
Built on organs (its numbers come first). Neighbours see only means.

  Distance. Every cell keeps its distance in steps from the channel's sink: the sink 0, others one more than the
  smallest of their neighbours' (from the mean of exp(-5 d), rounded). So a cell's neighbours are one step nearer,
  level, or one step farther.
  Which way. A cell shows its marks as 1, 9 or 81 by its distance mod 3. From the sum of its neighbours' it can then
  count those one step nearer, level and one step farther (three base-9 digits): it knows uphill from downhill.
  Growth. R and B are always on the path. A path cell with no path neighbour one step nearer is a tip. Its
  neighbours one step nearer offer to carry on the path, each with a random whole number; the tip shows the smallest
  offer it sees (and its distance mod 3, so they know it is one step farther); the cell whose offer it is joins. So
  a path grows one cell at a time. A path cell with no path neighbour one step farther (it lost its way back to a
  source) leaves it.
  Pulses. A path cell takes a pulse when a pulsing path neighbour is one step farther; so pulses only run down
  towards the sink. It carries it ON updates, then rests REST more. The sink counts the pulses it takes (an output).
"""

import numpy as np
from numba import njit

from rules import organs

FAR = organs.FAR
D_BETA = 5.0        # distances: sharp enough that the mean finds the smallest to within log 8 / 5 < 1/2
OFFERS = 99         # offers are whole numbers 1..OFFERS (a tie, two joining, is rare)
O_BETA = 5.0        # offers: the mean finds the smallest
T_BETA = 2.0        # a tip shows 3 x (the smallest offer, or OFFERS + 1 if none) + its distance mod 3
ON, REST = 6, 7     # updates a pulse is carried, then rested

CH = ("dist", "path", "tip", "pulse", "offer", "timer", "fired")
VARS = organs.VARS + tuple(f"{v}_{c}" for c in "av" for v in CH)
SHARED = organs.SHARED + tuple(x for c in "av" for x in ((f"dist_{c}", D_BETA), f"path_{c}", (f"tip_{c}", T_BETA), f"pulse_{c}",
                                                               (f"offer_{c}", O_BETA)))
INPUTS = ("send_a", "send_v")
OUTPUTS = ("fired_a", "fired_v")
BIRTH = {**organs.BIRTH, **{f"{v}_{c}": (FAR if v in ("dist", "tip", "offer") else 0.0) for c in "av" for v in CH}}
BIRTH_NOISE = organs.BIRTH_NOISE
BASE, M_BASE = len(organs.VARS), len(organs.SHARED)     # where each channel's numbers start: + 7 c, + 5 c
DIST, PATH, TIP, PULSE, OFFER, TIMER, FIRED = range(7)
SOURCES, SINKS = (1.0, 2.0), (3.0, 4.0)                 # names: R, B; A, V


@njit
def _digit(sum_, d):
    """How many neighbours a distance with this residue (mod 3) away show a mark, from the sum of what they show."""
    return (np.round(sum_) // 9 ** (d % 3)) % 9


@njit
def stage(me, mean, count, out, ch, send):
    """One channel: my distance, whether I am on the path (or a tip), my pulse. send: start a pulse, if I am a source."""
    v, m = BASE + 7 * ch, M_BASE + 5 * ch
    name = np.round(out[organs.NAME])
    source, sink = name == SOURCES[0] or name == SOURCES[1], name == SINKS[ch]
    d = 0.0 if sink else min(np.round(organs._smallest(count * mean[m], D_BETA)) + 1.0, FAR)
    out[v + DIST] = d
    known = d < FAR / 2
    mark = 9.0 ** (d % 3) if known else 0.0
    paths, pulses = count * mean[m + 1], count * mean[m + 3]
    tip = np.round(organs._smallest(count * mean[m + 2], T_BETA))   # a tip next to me: 3 x its choice + its d mod 3
    offers = organs._smallest(count * mean[m + 4], O_BETA)
    on, offer = me[v + PATH] > 0.5, np.round(me[v + OFFER])
    below = known and tip < FAR / 2 and tip % 3 == (d + 1) % 3       # a tip one step farther
    if source:
        on = known
    elif on:
        on = known and _digit(paths, d + 1) > 0                      # still a way back to a source
    elif below:
        if tip // 3 == offer:
            on = True                                               # it chose me
        elif offer > OFFERS:
            offer = float(np.random.randint(1, OFFERS + 1))
    out[v + OFFER] = offer if below and not on else FAR
    out[v + PATH] = mark if on else 0.0
    if on and d > 0 and _digit(paths, d - 1) == 0:
        out[v + TIP] = 3.0 * (np.round(offers) if offers <= OFFERS else OFFERS + 1.0) + d % 3
    else:
        out[v + TIP] = FAR
    timer = max(np.round(me[v + TIMER]) - 1.0, 0.0)
    fired = np.round(me[v + FIRED])
    if on and timer == 0 and ((source and send) or _digit(pulses, d + 1) > 0):
        timer = ON + REST
        if sink:
            fired += 1.0
    out[v + TIMER] = timer
    out[v + PULSE] = mark if on and timer > REST else 0.0
    out[v + FIRED] = fired


@njit
def update(me, mean, count, inp, out):
    organs.update(me, mean, count, inp, out)
    stage(me, mean, count, out, 0, inp[0] > 0.5)
    stage(me, mean, count, out, 1, inp[1] > 0.5)
