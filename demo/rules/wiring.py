"""Wiring, from averages only: paths grow from R and B to the tail, A (channel 0), and to its corner, V (channel 1), and
pulses run down them. Built on organs (its numbers come first). Neighbours see only means.

  Distance. A cell's distance to the tail is its row, y (from organs); to the corner at x = 0, y = 0, the larger of
  x and y. So a cell's
  neighbours are one step nearer, level, or one step farther.
  Which way. A cell shows its marks as 1, 9 or 81 by its distance mod 3. From the sum of its neighbours' it can then
  count those one step nearer, level and one step farther (three base-9 digits): it knows uphill from downhill.
  Growth. R and B are always on the path. A path cell with no path neighbour one step nearer is a tip. Its
  neighbours one step nearer offer to carry on the path, each with a random whole number; the tip shows the smallest
  offer it sees (and its distance mod 3, so they know it is one step farther); the cell whose offer it is joins. If two
  share the smallest (the mean shows it), the tip asks for new offers. A tip keeps its choice for LEASE updates (so it
  can't change its mind while the chosen cell is joining, and get two). A cell that would also be next to another
  path cell one step farther offers a larger number, so is chosen only if no other cell offers (joining would give
  that other cell a second way on, and its pulses would split in two). So
  a path grows one cell at a time. A path cell with no path neighbour one step farther (it lost its way back to a
  source) leaves it.
  Pulses. A path cell takes a pulse when a pulsing path neighbour is one step farther; so pulses only run down
  towards the sink. It carries it ON updates, then rests REST more. A sink cell counts the pulses it takes (an
  output: the body fires that thruster).
"""

import numpy as np
from numba import njit

from rules import organs

FAR = organs.FAR
OFFERS = 99         # offers are whole numbers 1..99, in three bands: 1..33 from a cell on the straight way, 34..66 from
BAND = 33           # one off it, 67..99 from one that would touch another path cell above it (and give it a second way
                    # on: pulses would split)
O_BETA = 5.0        # offers: the mean finds the smallest ...
LEASE = 6           # updates a tip keeps its choice
TIE = 0.07          # ... and if two share it, it comes out log 2 / 5 = 0.14 less than a whole number
T_BETA = 0.7        # a tip shows 9 x (the smallest offer, or OFFERS + 1 if none) + 3 x (x mod 3) + (y mod 3)
ON, REST = 6, 7     # updates a pulse is carried, then rested

CH = ("path", "tip", "pulse", "offer", "timer", "fired", "lease")
VARS = organs.VARS + tuple(f"{v}_{c}" for c in "av" for v in CH)
SHARED = organs.SHARED + tuple(x for c in "av" for x in (f"path_{c}", (f"tip_{c}", T_BETA), f"pulse_{c}",
                                                               (f"offer_{c}", O_BETA)))
INPUTS = ("send_a", "send_v")
OUTPUTS = ("fired_a", "fired_v")
BIRTH = {**organs.BIRTH, **{f"{v}_{c}": (FAR if v in ("tip", "offer") else 0.0) for c in "av" for v in CH}}
BIRTH_NOISE = organs.BIRTH_NOISE
BASE, M_BASE = len(organs.VARS), len(organs.SHARED)     # where each channel's numbers start: + 7 c, + 4 c
PATH, TIP, PULSE, OFFER, TIMER, FIRED, LEASE = range(7)
SOURCES, SINKS = (organs.R, organs.B), (organs.A, organs.V)


@njit
def _digit(sum_, d):
    """How many neighbours a distance with this residue (mod 3) away show a mark, from the sum of what they show."""
    return (np.round(sum_) // 9 ** (d % 3)) % 9


@njit
def stage(me, mean, count, out, ch, send):
    """One channel: whether I am on the path (or a tip), my pulse. send: start a pulse, if I am a source."""
    v, m = BASE + 7 * ch, M_BASE + 4 * ch
    name = np.round(out[organs.NAME])
    source, sink = name == SOURCES[0] or name == SOURCES[1], name == SINKS[ch]
    d = out[organs.Y] if ch == 0 else max(out[organs.X], out[organs.Y])
    known = d < FAR / 2
    mark = 9.0 ** (d % 3) if known else 0.0
    paths, pulses = count * mean[m], count * mean[m + 2]
    tip = np.round(organs._smallest(count * mean[m + 1], T_BETA))   # a tip next to me: its choice, x and y mod 3
    offers = organs._smallest(count * mean[m + 3], O_BETA)
    on, offer = me[v + PATH] > 0.5, np.round(me[v + OFFER])
    x, y = out[organs.X], out[organs.Y]
    below, choice, tx, ty = False, 0.0, FAR, FAR
    if known and tip < FAR / 2:
        choice = tip // 9
        tx = x + (((tip // 3) % 3 - x % 3 + 1) % 3 - 1)            # the tip's x and y: mine, one either way
        ty = y + ((tip % 3 - y % 3 + 1) % 3 - 1)
        below = (ty if ch == 0 else max(tx, ty)) == d + 1           # a tip one step farther
    if source:
        on = known
    elif on:
        on = known and _digit(paths, d + 1) > 0                      # still a way back to a source
    elif below:
        if choice == offer:
            on = True                                               # it chose me
        elif offer > OFFERS or choice == 0:                         # (0: a tie; offer again)
            if _digit(paths, d + 1) > 1:
                band = 2.0                                          # another path cell above me
            elif ch == 0:
                band = 0.0 if x == tx else 1.0                      # to the tail: straight down
            else:                                                   # to the corner: straight at it, the diagonal step
                far, near = max(tx, ty), min(tx, ty)                # taken in proportion
                diagonal = x < tx and y < ty
                band = 0.0 if diagonal == (np.random.random() < near / far) else 1.0
            offer = band * BAND + float(np.random.randint(1, BAND + 1))
    out[v + OFFER] = offer if below and not on else FAR
    out[v + PATH] = mark if on else 0.0
    lease = max(np.round(me[v + LEASE]) - 1.0, 0.0)
    if on and d > 0 and _digit(paths, d - 1) == 0:
        choice = OFFERS + 1.0                                       # none yet
        if lease > 0 and me[v + TIP] < FAR / 2:
            choice = np.round(me[v + TIP]) // 9                     # still holding my choice
        elif offers <= OFFERS:
            choice = np.round(offers)
            if choice - offers > TIE:
                choice = 0.0                                        # two share the smallest: ask again
            else:
                lease = LEASE
        out[v + TIP] = 9.0 * choice + 3.0 * (out[organs.X] % 3) + out[organs.Y] % 3
    else:
        out[v + TIP], lease = FAR, 0.0
    out[v + LEASE] = lease
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
