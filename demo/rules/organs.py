"""Where am I, from averages only: every cell comes to know its row and column, and the sheet names its organs: R and B
(two cells on the head's edge), A (the tail's edge) and V (a tail corner). Neighbours see only means.

  Every number a cell keeps is a whole number, re-rounded at each update, so the noise can't move it.
  1. Election. Corners (3 neighbours) put up a whole-number tag (random from birth, 0..139); every cell holds the
     smallest it has heard: the smallest of its neighbours', found from the mean of exp(-5 x least) and rounded. The
     corner whose own tag survives is the origin.
  2. Handedness. The origin shows 1 - 2 x the sum of what its neighbours show (count x the mean). An edge cell (5
     neighbours) that sees a positive sum is next to the origin: it shows minus a second tag of its own, so the
     origin shows 1 + 2 x both tags; the two are also each other's neighbours, so each sees 1 + 2 x its own + the
     other's, and so learns the other's. The smaller starts ring
     count a, the other ring count b, and each blocks the other's count (as the origin blocks both).
  3. Ring counts. Edge cells (fewer than 8 neighbours) count steps round the edge: one more than the smallest of
     their neighbours' (inner cells show nothing). So an edge cell knows a, b and the perimeter a + b, and so its side
     (a quarter of the perimeter each).
  4. Coordinates. The first side's cells hold y = 0, every other cell one more than its neighbours' smallest (a corner
     touching that side, 0): with 8 neighbours that is exactly the row. The next side round counts x likewise.
  5. Naming. The first side (y = 0) is the tail, A, bar its corner at x = 0, V. Two cells on the far side, the head,
     name themselves R and B: their place round the edge is 5/8 of the perimeter, 3 cells either way. A cell takes a
     name once it has held it for HOLD updates in a row: at first every corner thinks it won the election, and the
     losers' counts take a while to rise away, giving places that drift as they rise, so never hold.
"""

import numpy as np
from numba import njit

FAR = 1e9
TAGS = 139          # election tags are whole numbers 0..TAGS (exp(-5 x 139) is still a number)
E_BETA = 5.0        # the mean of exp(-5 v) finds the smallest v to within log 8 / 5 < 1/2
R_BETA = 1.5        # ring counts: an edge cell has one neighbour nearer round the ring, so the sum is sharp
HOLD = 80           # updates a cell must hold a name before it takes it
SENSORS = np.array([(0.625, -3.0), (0.625, 3.0)])   # R, B: (fraction of the perimeter from the origin, offset in cells)
R, B, A, V = 1.0, 2.0, 3.0, 4.0                     # names

VARS = ("tag", "tag2", "least", "origin", "hand", "role", "a", "b", "y", "x", "place", "held", "name")
SHARED = (("least", E_BETA), "hand", ("a", R_BETA), ("b", R_BETA), ("y", E_BETA), ("x", E_BETA))
INPUTS = ()
BIRTH = {"tag": 0.0, "tag2": 0.0, "least": FAR, "origin": 0.0, "hand": 0.0, "role": 0.0, "a": FAR, "b": FAR,
         "y": FAR, "x": FAR, "place": 0.0, "held": 0.0, "name": 0.0}
BIRTH_NOISE = {"tag": TAGS + 1.0, "tag2": 1e6}      # whole numbers once rounded
TAG, TAG2, LEAST, ORIGIN, HAND, ROLE, RA, RB, Y, X, PLACE, HELD, NAME = range(13)
M_LEAST, M_HAND, M_A, M_B, M_Y, M_X = range(6)


@njit
def _smallest(sum_, beta):
    """The smallest of the neighbours' values, from the sum of exp(-beta x value); FAR if none."""
    return -np.log(sum_) / beta if sum_ > 1e-300 else FAR


@njit
def update(me, mean, count, inp, out):
    corner, ring = count == 3, count < 8
    mine = min(np.round(me[TAG]), TAGS)                 # numbers a cell keeps are whole, so noise can't move them
    out[TAG], out[TAG2] = mine, np.round(me[TAG2])
    # 1. election (the mean finds the smallest to within log(count) / 5: rounding takes it to the whole number)
    least = np.round(min(me[LEAST], _smallest(mean[M_LEAST], E_BETA)))
    if corner:
        least = min(least, mine)
    out[LEAST] = least
    origin = corner and least == mine
    out[ORIGIN] = 1.0 if origin else 0.0
    # 2. handedness
    hands = np.round(count * mean[M_HAND])
    candidate = count == 5 and hands > 0.5
    role = 0.0
    if origin:
        out[HAND] = 1.0 - 2.0 * hands                   # 1 + 2 x both tags (they show minus theirs)
    elif candidate:
        out[HAND] = -out[TAG2]
        role = 1.0 if out[TAG2] < hands - 1.0 - 2.0 * out[TAG2] else -1.0
    else:
        out[HAND] = 0.0
    out[ROLE] = role
    # 3. ring counts
    for v, m, seed in ((RA, M_A, 1.0), (RB, M_B, -1.0)):
        if role == seed:
            out[v] = 1.0
        elif not ring or origin or role == -seed:
            out[v] = FAR
        else:
            out[v] = min(np.round(_smallest(count * mean[m], R_BETA)) + 1.0, FAR)
    counted = count == 5 and out[RA] < FAR / 2 and out[RB] < FAR / 2
    side = -1.0
    if role == 1.0:
        side = 0.0                                      # where count a starts: the first side
    elif counted:
        side = np.floor(4.0 * out[RA] / (out[RA] + out[RB]))
    # 4. coordinates
    for v, m, k in ((Y, M_Y, 0.0), (X, M_X, 1.0)):
        heard = np.round(_smallest(count * mean[m], E_BETA))
        out[v] = 0.0 if side == k or (corner and heard == 0.0) else min(heard + 1.0, FAR)
    # 5. naming
    place = 0.0
    if out[Y] == 0.0 and out[X] == 0.0:
        place = V
    elif out[Y] == 0.0:
        place = A
    elif counted:
        for k in range(SENSORS.shape[0]):
            if out[RA] == np.round(SENSORS[k, 0] * (out[RA] + out[RB])) + SENSORS[k, 1]:
                place = k + 1.0
    held = min(np.round(me[HELD]) + 1.0, HOLD) if place == np.round(me[PLACE]) else 0.0
    out[PLACE], out[HELD] = place, held
    out[NAME] = place if held >= HOLD else 0.0
