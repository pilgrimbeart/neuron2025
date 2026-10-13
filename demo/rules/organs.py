"""Where am I, from averages only: the sheet's edge cells come to know where they sit round the edge, and four of them
name themselves organs (R, B, A, V). Neighbours see only means.

  Every number a cell keeps is a whole number, re-rounded at each update, so the noise can't move it.
  1. Election. Corners (3 neighbours) put up a whole-number tag (random from birth, 0..139); every cell holds the
     smallest it has heard: the smallest of its neighbours', found from the mean of exp(-5 x least) and rounded. The
     corner whose own tag survives is the origin.
  2. Handedness. The origin's two edge neighbours (5 neighbours, one of them the origin) each show a second tag; the
     origin shows the sum it sees (3 x the mean); each subtracts its own to learn the other's. The smaller starts
     ring count a, the other ring count b, and each blocks the other's count (as the origin blocks both).
  3. Ring counts. Edge cells (fewer than 8 neighbours) count steps round the edge: one more than the smallest of
     their neighbours' (inner cells show nothing). So an edge cell knows a, b and the perimeter a + b.
  4. Naming. An edge cell (not a corner) whose a is an organ's place, a fraction of the perimeter plus an offset in
     cells, is at that place; once it has been there for HOLD updates in a row, it takes the organ's name. (At first
     every corner thinks it won the election, and the losers' ring counts take a while to rise away; the places they
     give drift along the edge as they rise, so never hold, and nothing is wired to them.)
  5. Coordinates. An edge cell knows its side from its place round the ring (a quarter of the perimeter each). The
     side A is on counts rows in: its cells hold y = 0, every other cell one more than its neighbours' smallest (a
     corner touching that side, 0). With 8 neighbours that is exactly the row. The next side round counts x
     likewise. (Nothing uses them yet but the watcher's checkerboard.)
"""

import numpy as np
from numba import njit

FAR = 1e9
TAGS = 139          # election tags are whole numbers 0..TAGS (exp(-5 x 139) is still a number)
E_BETA = 5.0        # election: sharp enough that the mean finds the smallest tag to within log 8 / 5 < 1/2
R_BETA = 1.5        # ring counts: an edge cell has one neighbour nearer round the ring, so the sum is sharp
HOLD = 80           # updates a cell must be at an organ's place before it takes the name
# Organs: (fraction of the perimeter from the origin, offset in cells). a runs round: origin 0, first corner 1/4,
# far corner 1/2, ... R and B on the far side's middle (5/8), 6 cells apart; A on the near side's middle (1/8); V on
# the middle of a side between them (3/8).
ORGANS = np.array([(0.625, -3.0), (0.625, 3.0), (0.125, 0.0), (0.375, 0.0)])

VARS = ("tag", "tag2", "least", "origin", "q", "p", "role", "a", "b", "name", "place", "held", "y", "x")
SHARED = (("least", E_BETA), "origin", "q", "p", ("a", R_BETA), ("b", R_BETA), ("y", E_BETA), ("x", E_BETA))
INPUTS = ()
BIRTH = {"tag": 0.0, "tag2": 0.0, "least": FAR, "origin": 0.0, "q": 0.0, "p": 0.0, "role": 0.0, "a": FAR, "b": FAR, "name": 0.0,
         "place": 0.0, "held": 0.0, "y": FAR, "x": FAR}
BIRTH_NOISE = {"tag": TAGS + 1.0, "tag2": 1e6}      # whole numbers once rounded
TAG, TAG2, LEAST, ORIGIN, Q, P, ROLE, A, B, NAME, PLACE, HELD, Y, X = range(14)
M_LEAST, M_ORIGIN, M_Q, M_P, M_A, M_B, M_Y, M_X = range(8)


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
    candidate = count == 5 and mean[M_ORIGIN] > 0.1
    out[Q] = out[TAG2] if candidate else 0.0
    out[P] = count * mean[M_Q] if origin else 0.0
    role = 0.0
    if candidate:
        other = np.round(count * mean[M_P] - out[TAG2])
        role = 1.0 if out[TAG2] < other else -1.0
    out[ROLE] = role
    # 3. ring counts
    for v, m, seed in ((A, M_A, 1.0), (B, M_B, -1.0)):
        if role == seed:
            out[v] = 1.0
        elif not ring or origin or role == -seed:
            out[v] = FAR
        else:
            out[v] = np.round(_smallest(count * mean[m], R_BETA)) + 1.0
            out[v] = min(out[v], FAR)
    # 4. naming
    place = 0.0
    if count == 5 and out[A] < FAR / 2 and out[B] < FAR / 2:
        perimeter = out[A] + out[B]
        for k in range(ORGANS.shape[0]):
            if out[A] == np.round(ORGANS[k, 0] * perimeter) + ORGANS[k, 1]:
                place = k + 1.0
    held = min(np.round(me[HELD]) + 1.0, HOLD) if place == np.round(me[PLACE]) else 0.0
    out[PLACE], out[HELD] = place, held
    out[NAME] = place if held >= HOLD else 0.0
    # 5. coordinates
    side = -1.0
    if role == 1.0:
        side = 0.0                                          # where count a starts: the first side
    elif count == 5 and out[A] < FAR / 2 and out[B] < FAR / 2:
        side = np.floor(4.0 * out[A] / (out[A] + out[B]))
    for v, m, k in ((Y, M_Y, 0.0), (X, M_X, 1.0)):
        heard = np.round(_smallest(count * mean[m], E_BETA))
        if side == k or (corner and heard == 0.0):
            out[v] = 0.0
        else:
            out[v] = min(heard + 1.0, FAR)
