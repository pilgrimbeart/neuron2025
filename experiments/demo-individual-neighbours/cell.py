"""The cell: everything a cell is and does. Every cell of the sheet holds the same numbers and runs this rule.

A cell sees only its own numbers and what its (up to 8) neighbours share. It knows nothing of the sheet's size, of its
position, of a clock or of a compass. All cells are born the same, bar random noise (born). Each tick every cell's
timers run, and then each cell, with chance UPDATE, updates from its own numbers and its neighbours' (tick): never
all at once.

Its numbers come in four layers, which the sheet builds up in stages, each starting when the one before has produced
what it needs (no timers between them):

  1. Where am I (the frame). Corners know they are corners (three neighbours). Every cell holds the smallest corner
     tag it has heard: the corner whose own tag survives is the origin, (0, 0). Its two edge neighbours choose the
     axes by their tags, and the edges get labelled by zeros in four whole-number counts, in from the south, west,
     east and north edges. Every other count is the smallest neighbour's plus one, so x = west / (west + east) and y
     = south / (south + north). A cell whose coordinates agree with its neighbours' names itself an organ if it is
     nearer that organ's place (ORGANS) than any neighbour: R and B on the north edge, A at the south, V near the
     north-east corner.
  2. Wiring (two channels of paths, one per target: A, and V). A cell named A (V) is a sink in channel 0 (1); R and
     B are sources in both. A hungry sink sends out waves that count their steps; each cell remembers which
     neighbour the wave came from (its breadcrumb, by that neighbour's tag); a path is the trail of breadcrumbs from
     a source. Pulses run down the path, the sink thanks each one with an acknowledgement back up it, and a thanked
     path stays while the sheet falls quiet.
  3. Choosing (the input cells, R and B). A sensor kick makes the cell send a pulse towards A with chance its
     balance b, else towards V.
  4. Learning. Every cell keeps how long since a sensor kicked it (its trace fades from 1) and the news: the age of
     the freshest kick it has heard of, its own or a neighbour's (plus a little per step), spreading like a chemical.
     A cell that hears of a kick fresher than its own forgets its own (lateral inhibition), so of R and B only the
     more recently kicked keeps a trace. A taste moves b by RATE x taste x trace
     (three factors: just active, a meal, its taste), and a bad one makes the cell send towards V for a while:
     disgust.

A cell reads of its neighbours only what they show (show): 18 numbers, listed there. It also knows which of its
neighbours touch it by a side and which by a corner, and which of them touch each other.
"""

from __future__ import annotations

import numpy as np
from numba import njit

UPDATE = 0.9        # the chance that a cell updates in a tick (never all at once)
FAR = 1e9           # nothing heard yet

# Where am I
ORGANS = np.array([(0.5, 1.0, -3, 0),  # R: the north edge's middle, 3 cells west (x, y as fractions, plus cells)
                   (0.5, 1.0, 3, 0),   # B: 3 cells east
                   (0.5, 0.0, 0, 0),   # A: the south edge's middle
                   (0.8, 0.8, 0, 0)])  # V: towards the north-east corner
R, B, A, V = 1, 2, 3, 4                # names (0: none)
COUNT_NOISE = 0.001 # random jitter in the counts at each update (whole numbers wipe it out)
TIE = 1e-4          # how much nearer a neighbour must be to a place before a cell gives up its name

# Wiring
SPONTANEOUS = 0.5   # how often an end acts by itself (a hungry sink sends a wave, a lonely source asks)
EXCITED = 16.0      # how long a cell carries a wave
REFRACTORY = 40.0   # how long it then rests (longer, so a wave can't turn back)
EXPIRE = 200.0      # a cell keeps its breadcrumb this long without a wave or thanks
WAVE_NOISE = 0.01   # random jitter in each count a wave carries
MARGIN = 0.1        # how much longer the way through its breadcrumb may be before a cell switches
PULSE_TIME = 16.0   # how long a cell passes on a pulse or thanks
PULSE_REST = 2.0    # how long it then rests from passing another
HUNGER = 200.0      # a sink sends waves after this long without a pulse; a source asks after this long unthanked
CALL = 150.0        # a sink a request reaches keeps calling this long
KEEP = 50.0         # a cell thanked within this long keeps its breadcrumb whatever waves pass

# Choosing and learning
INNATE = 0.9        # the balance a cell is born with (drawn to anything it sees) ...
BALANCE_NOISE = 0.02  # ... give or take this much
TRACE = 30.0        # how long a trace lasts
RATE = 1.0          # how far one taste moves the balance
DISGUST = 40.0      # how long a bad taste lasts
HOP = 0.1           # news ages this much more per step it travels (so a cell's own echo is never fresher)
RECOIL = 0.2        # how often a disgusted cell sends a pulse towards V (per tick)

# The layers: each a number every cell holds.
# Frame: the smallest corner tag heard, the four counts, x, y, the name. Shared: TAG, LEAST, the counts.
TAG, LEAST, D_S, D_W, D_E, D_N, X, Y, NAME = range(9)
# Learning: balance, ticks since kicked, a taste delivered and not yet taken, disgust left, the news (shared).
BALANCE, KICKED, TASTE, DISGUSTED, NEWS = range(9, 14)
# A path channel: the wave (excited and resting time left, its count), the breadcrumb and what the cell watched
# while excited (the count via its breadcrumb, the best count and whose), ticks since a wave or thanks, whether on
# the path; a pulse passing (time left, rest left), thanks passing (the same), ticks since a sink was fed, pulses it
# received; a request passing (time left, rest left), ticks since a source was thanked, how long a sink keeps
# calling, ticks since thanked.
(WAVE, WAVE_REST, COUNT, CRUMB, VIA, BEST, BEST_CRUMB, AGE, PATH, PULSE, PULSE_R, ACK, ACK_R, FED, ARRIVED,
 ASK, ASK_R, LONELY, CALLING, THANKED) = range(20)
CHANNEL = (14, 34)  # where each channel's layers start (+ the indices above): to A, to V
LAYERS = 54

# What a cell shows its neighbours: all they can read of it (rebuilt each tick from its layers, by show).
# Its tag, the smallest corner tag it has heard, its four counts and how many neighbours it has (frame); per channel
# its wave's count (while it carries a wave, else -1), its breadcrumb (while on the path, else -1), and whether it is
# passing a pulse, a request or thanks; and its news. 18 numbers.
SH_TAG, SH_LEAST, SH_D, SH_KIND = 0, 1, 2, 6         # SH_D: four counts, SH_D + 0..3
SH_CHANNEL = (7, 12)                                # + SH_COUNT ... SH_ACK
SH_COUNT, SH_CRUMB, SH_PULSE, SH_ASK, SH_ACK = range(5)
SH_NEWS = 17
SHOWN = 18


def born(n: int, rng) -> np.ndarray:
    """A new sheet, n x n: every cell the same, bar its random tag, the noise in x and y, and in its balance."""
    s = np.zeros((LAYERS, n, n))
    s[TAG] = rng.random((n, n))                     # a name from birth: nothing about position
    s[LEAST] = 2.0                                  # no corner tag heard yet
    s[D_S:D_N + 1] = FAR                            # no edge heard from yet
    s[X], s[Y] = rng.uniform(0, 0.1, (n, n)), rng.uniform(0, 0.1, (n, n))
    s[BALANCE] = INNATE + BALANCE_NOISE * rng.standard_normal((n, n))
    s[KICKED] = s[NEWS] = FAR
    for c in CHANNEL:
        s[c + COUNT] = s[c + VIA] = s[c + BEST] = FAR
        s[c + CRUMB] = s[c + BEST_CRUMB] = -1.0
        s[c + AGE] = EXPIRE + 1.0
        s[c + FED] = s[c + LONELY] = s[c + THANKED] = FAR   # sinks born hungry, sources lonely
    return s


NOISE_LAYERS = (TAG, X, Y, BALANCE)                 # the layers that differ between cells at birth


@njit(cache=True)
def seed(n):
    np.random.seed(n)


@njit(cache=True)
def _neighbours(n, y, x):
    """How many neighbours a cell has: 3 at a corner, 5 on an edge, else 8."""
    return (3 if 0 < y < n - 1 else 2) * (3 if 0 < x < n - 1 else 2) - 1


@njit(cache=True)
def _zero(v):
    return abs(v) < 0.5


@njit(cache=True)
def _role(name, channel):
    """Source (1), sink (2) or neither (0) in a channel, by the cell's name."""
    if name == R or name == B:
        return 1
    return 2 if name == (A if channel == 0 else V) else 0


@njit(cache=True)
def tick(s, s2, kick, taste, learning):
    """One tick. kick marks the cells a sensor kicks now, taste is the taste arriving at each cell now (0 almost
    everywhere); both come from the body, at cells it found by their names. learning False holds every balance
    fixed (for comparison)."""
    n = s.shape[1]
    root2 = np.sqrt(2.0)
    # Every cell, every tick: inputs arrive, timers run, kicked cells send pulses, ends act now and then.
    for y in range(n):
        for x in range(n):
            s[TASTE, y, x] += taste[y, x]
            s[KICKED, y, x] = min(FAR, s[KICKED, y, x] + 1.0)
            s[NEWS, y, x] = min(FAR, s[NEWS, y, x] + 1.0)
            send = np.zeros(2, np.bool_)                            # a pulse into each channel
            if kick[y, x]:                                          # choose: a pulse towards A or V; my news is
                s[KICKED, y, x] = s[NEWS, y, x] = 0.0               # fresh
                send[0 if np.random.random() < s[BALANCE, y, x] else 1] = True
            if s[DISGUSTED, y, x] > 0.0:                            # disgust: recoil
                s[DISGUSTED, y, x] = max(0.0, s[DISGUSTED, y, x] - 1.0)
                if np.random.random() < RECOIL:
                    send[1] = True
            for ch in range(2):
                c = CHANNEL[ch]
                role = _role(s[NAME, y, x], ch)
                s[c + AGE, y, x] += 1.0
                if s[c + WAVE, y, x] > 0.0:
                    s[c + WAVE, y, x] -= 1.0
                    if s[c + WAVE, y, x] <= 0.0:                    # the wave has passed: rest, and settle the
                        s[c + WAVE, y, x] = 0.0                     # breadcrumb (kept unless the best way was
                        s[c + WAVE_REST, y, x] = REFRACTORY         # clearly shorter, or it is in use)
                        if (s[c + VIA, y, x] > s[c + BEST, y, x] * (1.0 + MARGIN) and s[c + BEST_CRUMB, y, x] >= 0
                                and role != 2 and s[c + THANKED, y, x] > KEEP):
                            s[c + CRUMB, y, x] = s[c + BEST_CRUMB, y, x]
                elif s[c + WAVE_REST, y, x] > 0.0:
                    s[c + WAVE_REST, y, x] = max(0.0, s[c + WAVE_REST, y, x] - 1.0)
                s[c + FED, y, x] += 1.0
                s[c + LONELY, y, x] += 1.0
                s[c + THANKED, y, x] += 1.0
                s[c + CALLING, y, x] = max(0.0, s[c + CALLING, y, x] - 1.0)
                for left, rest, wait in ((ASK, ASK_R, REFRACTORY), (PULSE, PULSE_R, PULSE_REST),
                                         (ACK, ACK_R, PULSE_REST)):
                    if s[c + left, y, x] > 0.0:
                        s[c + left, y, x] -= 1.0
                        if s[c + left, y, x] <= 0.0:
                            s[c + left, y, x], s[c + rest, y, x] = 0.0, wait
                    elif s[c + rest, y, x] > 0.0:
                        s[c + rest, y, x] = max(0.0, s[c + rest, y, x] - 1.0)
                if (send[ch] and role == 1 and s[c + PATH, y, x] > 0.5
                        and s[c + PULSE, y, x] == 0.0 and s[c + PULSE_R, y, x] == 0.0):
                    s[c + PULSE, y, x] = PULSE_TIME                  # the pulse sets off down the path
                if role == 1 and np.random.random() < SPONTANEOUS and s[c + LONELY, y, x] > HUNGER \
                        and s[c + ASK, y, x] == 0.0 and s[c + ASK_R, y, x] == 0.0:
                    s[c + ASK, y, x] = EXCITED                      # a lonely source asks to be wired
                if role == 2 and np.random.random() < SPONTANEOUS and s[c + WAVE, y, x] == 0.0 \
                        and s[c + WAVE_REST, y, x] == 0.0 and (s[c + FED, y, x] > HUNGER or s[c + CALLING, y, x] > 0):
                    s[c + WAVE, y, x], s[c + COUNT, y, x], s[c + AGE, y, x] = EXCITED, 0.0, 0.0   # a hungry sink
                    s[c + CRUMB, y, x], s[c + VIA, y, x] = -1.0, 0.0                              # calls
                    s[c + BEST, y, x], s[c + BEST_CRUMB, y, x] = FAR, -1.0
    # Each cell, with chance UPDATE, from its own numbers and what its neighbours show.
    sh = show(s)
    s2[:, :, :] = s
    for y in range(n):
        for x in range(n):
            if np.random.random() >= UPDATE:
                continue
            _frame(s, s2, sh, n, y, x)
            for ch in range(2):
                _path(s, s2, sh, n, y, x, ch, _role(s[NAME, y, x], ch), root2)
            _learn(s, s2, sh, n, y, x, learning)
    s[:, :, :] = s2


@njit(cache=True)
def show(s):
    """What every cell shows its neighbours: the only thing a cell can read of another."""
    n = s.shape[1]
    sh = np.empty((SHOWN, n, n))
    for y in range(n):
        for x in range(n):
            sh[SH_TAG, y, x], sh[SH_LEAST, y, x] = s[TAG, y, x], s[LEAST, y, x]
            for e in range(4):
                sh[SH_D + e, y, x] = s[D_S + e, y, x]
            sh[SH_KIND, y, x] = _neighbours(n, y, x)
            for ch in range(2):
                c, k = CHANNEL[ch], SH_CHANNEL[ch]
                sh[k + SH_COUNT, y, x] = s[c + COUNT, y, x] if s[c + WAVE, y, x] > 0.0 else -1.0
                sh[k + SH_CRUMB, y, x] = s[c + CRUMB, y, x] if s[c + PATH, y, x] > 0.5 else -1.0
                sh[k + SH_PULSE, y, x] = 1.0 if s[c + PULSE, y, x] > 0.0 else 0.0
                sh[k + SH_ASK, y, x] = 1.0 if s[c + ASK, y, x] > 0.0 else 0.0
                sh[k + SH_ACK, y, x] = 1.0 if s[c + ACK, y, x] > 0.0 else 0.0
            sh[SH_NEWS, y, x] = s[NEWS, y, x]
    return sh


@njit(cache=True)
def _frame(s, s2, sh, n, y, x):
    """Where am I: the election, the edges' zeros, the counts, x and y, and naming."""
    kind = _neighbours(n, y, x)
    corner, edge = kind == 3, kind == 5
    least = s[TAG, y, x] if corner else 2.0                 # the election: the smallest corner tag heard
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yy, xx = y + dy, x + dx
            if (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                least = min(least, sh[SH_LEAST, yy, xx])
    s2[LEAST, y, x] = least
    fresh = s[LEAST, y, x] != least                         # a new winner believed in: start again under it
    zero = np.zeros(4)                                      # which edges this cell is on (its counts that are 0)
    if not fresh:
        for e in range(4):
            zero[e] = 1.0 if _zero(s[D_S + e, y, x]) else 0.0
    labelled = zero.sum()
    near = np.full(4, FAR)                                  # the smallest neighbour's count + 1, same winner
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yy, xx = y + dy, x + dx
            if not ((dy or dx) and 0 <= yy < n and 0 <= xx < n) or sh[SH_LEAST, yy, xx] != least:
                continue
            for e in range(4):
                near[e] = min(near[e], np.round(sh[SH_D + e, yy, xx]) + 1.0)
            other = sh[SH_KIND, yy, xx]
            if not (edge or corner) or (dy and dx) or other == 8:
                continue                                    # below: edge cells and corners, along the edge
            if other == 5:                                  # an edge neighbour's zero runs on along the edge
                for e in range(4):
                    if _zero(sh[SH_D + e, yy, xx]):
                        zero[e] = 1.0
            elif sh[SH_TAG, yy, xx] == least:                   # beside the origin: of its two edge neighbours the
                for ey in range(-1, 2):                     # smaller tag takes the south edge, the other the west
                    for ex in range(-1, 2):
                        oy, ox = y + ey, x + ex
                        if ey and ex and 0 <= oy < n and 0 <= ox < n and sh[SH_KIND, oy, ox] == 5 \
                                and abs(oy - yy) + abs(ox - xx) == 1:
                            zero[0 if s[TAG, y, x] < sh[SH_TAG, oy, ox] else 1] = 1.0
            elif labelled == 0.0:                           # beside another corner, no zero of my own yet: take
                for e in range(4):                          # its zeros my other edge neighbour lacks
                    if _zero(sh[SH_D + e, yy, xx]):
                        lacks = True
                        for ey in range(-1, 2):
                            for ex in range(-1, 2):
                                oy, ox = y + ey, x + ex
                                if ey and ex and 0 <= oy < n and 0 <= ox < n and sh[SH_KIND, oy, ox] == 5 \
                                        and abs(oy - yy) + abs(ox - xx) == 1 \
                                        and sh[SH_LEAST, oy, ox] == least and _zero(sh[SH_D + e, oy, ox]):
                                    lacks = False
                        if lacks:
                            zero[e] = 1.0
    if corner:                                              # which corner this is (edges: 0 S, 1 W, 2 E, 3 N)
        if least == s[TAG, y, x]:
            zero[0] = zero[1] = 1.0                         # the origin
        elif zero[0] > 0:
            zero[2] = 1.0                                   # (1, 0)
        elif zero[1] > 0:
            zero[3] = 1.0                                   # (0, 1)
        elif zero[2] > 0 or zero[3] > 0:
            zero[2] = zero[3] = 1.0                         # (1, 1)
    for e in range(4):                                      # count in from each edge
        s2[D_S + e, y, x] = (0.0 if zero[e] > 0 else near[e]) + COUNT_NOISE * np.random.standard_normal()
    ds, dw, de, dn = (np.round(s2[D_S, y, x]), np.round(s2[D_W, y, x]),
                      np.round(s2[D_E, y, x]), np.round(s2[D_N, y, x]))
    if dw + de < FAR / 2 and ds + dn < FAR / 2 and dw + de > 0.5 and ds + dn > 0.5:
        s2[X, y, x], s2[Y, y, x] = dw / (dw + de), ds / (ds + dn)
    # Naming: once this cell's sums of opposite counts equal its neighbours' (its frame is finished here), it takes
    # an organ's name if it is nearer the organ's place than any neighbour (on a tie the one to the west, or south).
    across = np.round(s[D_W, y, x]) + np.round(s[D_E, y, x])
    down = np.round(s[D_S, y, x]) + np.round(s[D_N, y, x])
    knows = across < FAR / 2 and down < FAR / 2 and across > 0.5 and down > 0.5 and not fresh
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yy, xx = y + dy, x + dx
            if knows and (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                knows = (sh[SH_LEAST, yy, xx] == least
                         and np.round(sh[SH_D + 1, yy, xx]) + np.round(sh[SH_D + 2, yy, xx]) == across
                         and np.round(sh[SH_D, yy, xx]) + np.round(sh[SH_D + 3, yy, xx]) == down)
    name = 0.0
    for p in range(ORGANS.shape[0] if knows else 0):
        tx = ORGANS[p, 0] + ORGANS[p, 2] / across           # a cell is 1 / width across: offsets in cells
        ty = ORGANS[p, 1] + ORGANS[p, 3] / down
        mx, my = np.round(s[D_W, y, x]) / across, np.round(s[D_S, y, x]) / down
        mine = abs(mx - tx) + abs(my - ty)
        nearest = True
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                yy, xx = y + dy, x + dx
                if (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                    ox, oy = np.round(sh[SH_D + 1, yy, xx]) / across, np.round(sh[SH_D, yy, xx]) / down
                    theirs = abs(ox - tx) + abs(oy - ty)
                    if theirs < mine - TIE or (theirs <= mine + TIE and (ox < mx or (ox == mx and oy < my))):
                        nearest = False
        if nearest:
            name = p + 1.0
    s2[NAME, y, x] = name


@njit(cache=True)
def _path(s, s2, sh, n, y, x, ch, role, root2):
    """Wiring, in one channel: waves and breadcrumbs, pulses down the path, thanks back up it, requests."""
    c, k = CHANNEL[ch], SH_CHANNEL[ch]
    best, best_crumb, via = FAR, -1.0, FAR      # the best way the wave comes (count + step, whose); via my crumb
    named = fired_at = acked = asked = False    # a path cell's crumb names me; it passes me a pulse; the cell my
    for dy in range(-1, 2):                     # crumb names passes back thanks; a neighbour passes a request
        for dx in range(-1, 2):
            yy, xx = y + dy, x + dx
            if not ((dy or dx) and 0 <= yy < n and 0 <= xx < n):
                continue
            m = sh[k + SH_COUNT, yy, xx]
            if m >= 0.0:                                                # in a wave
                v = m + (root2 if dy and dx else 1.0)                   # (a step across a corner is longer)
                if v < best:
                    best, best_crumb = v, sh[SH_TAG, yy, xx]
                if sh[SH_TAG, yy, xx] == s[c + CRUMB, y, x] and m < s[c + COUNT, y, x]:
                    via = v
            if sh[k + SH_CRUMB, yy, xx] == s[TAG, y, x]:                # a path cell whose breadcrumb names me
                named = True
                if sh[k + SH_PULSE, yy, xx] > 0.5:
                    fired_at = True
            if sh[SH_TAG, yy, xx] == s[c + CRUMB, y, x] and sh[k + SH_ACK, yy, xx] > 0.5:
                acked = True
            if sh[k + SH_ASK, yy, xx] > 0.5:
                asked = True
    if s[c + WAVE, y, x] > 0.0:                         # in a wave: take a smaller count, watch the ways it comes
        if best < s[c + COUNT, y, x] and s[c + COUNT, y, x] > 0.0:
            s2[c + COUNT, y, x] = best
        if via < s[c + VIA, y, x]:
            s2[c + VIA, y, x] = via
        if best < s[c + BEST, y, x]:
            s2[c + BEST, y, x], s2[c + BEST_CRUMB, y, x] = best, best_crumb
    elif s[c + WAVE_REST, y, x] == 0.0 and best < FAR:  # ready, next to a wave: carry it on
        s2[c + WAVE, y, x] = EXCITED
        s2[c + COUNT, y, x] = best + WAVE_NOISE * np.random.standard_normal()
        s2[c + AGE, y, x] = 0.0
        s2[c + VIA, y, x], s2[c + BEST, y, x], s2[c + BEST_CRUMB, y, x] = via, best, best_crumb
    if s[c + PATH, y, x] > 0.5:
        if fired_at and s[c + PULSE, y, x] == 0.0 and s[c + PULSE_R, y, x] == 0.0:
            s2[c + PULSE, y, x] = PULSE_TIME            # the pulse passes on
            if role == 2:                               # a sink fed: it thanks the path
                s2[c + FED, y, x], s2[c + ARRIVED, y, x] = 0.0, s[c + ARRIVED, y, x] + 1.0
                s2[c + ACK, y, x] = PULSE_TIME
        if acked and s[c + ACK, y, x] == 0.0 and s[c + ACK_R, y, x] == 0.0:
            s2[c + ACK, y, x] = PULSE_TIME              # thanks pass back up, keeping the breadcrumb (in use)
            s2[c + AGE, y, x] = s2[c + THANKED, y, x] = 0.0
            if role == 1:
                s2[c + LONELY, y, x] = 0.0
    if asked and s[c + ASK, y, x] == 0.0 and s[c + ASK_R, y, x] == 0.0:
        s2[c + ASK, y, x] = EXCITED                     # a request passes on; a sink it reaches calls a while
        if role == 2:
            s2[c + CALLING, y, x] = CALL
    if s[c + AGE, y, x] >= EXPIRE and s2[c + AGE, y, x] >= EXPIRE:
        s2[c + CRUMB, y, x] = -1.0                      # no wave or thanks for a long while: forget
    s2[c + PATH, y, x] = 1.0 if role == 1 or named else 0.0


@njit(cache=True)
def _learn(s, s2, sh, n, y, x, learning):
    """Learning: news of the freshest kick (a fresher one than my own wipes my trace), and a taste moves my balance."""
    news = s[NEWS, y, x]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yy, xx = y + dy, x + dx
            if (dy or dx) and 0 <= yy < n and 0 <= xx < n:
                news = min(news, sh[SH_NEWS, yy, xx] + HOP)
    s2[NEWS, y, x] = news
    if news < s[KICKED, y, x]:                          # someone was kicked after me: forget my kick
        s2[KICKED, y, x] = FAR
    t = s[TASTE, y, x]
    if t != 0.0:
        b = s[BALANCE, y, x]
        trace = np.exp(-s2[KICKED, y, x] / TRACE)
        if learning:
            s2[BALANCE, y, x] = b + RATE * t * trace * (b if t < 0.0 else 1.0 - b)
        s2[TASTE, y, x] = 0.0
        if t < 0.0:
            s2[DISGUSTED, y, x] = DISGUST
