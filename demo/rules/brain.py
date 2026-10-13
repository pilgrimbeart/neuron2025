"""The creature's brain, from averages only: organs, wiring, then choosing and learning. Built on wiring (its numbers
come first). Neighbours see only means.

  Choosing. When the body kicks R or B (its sensor saw that colour), that cell sends a pulse towards A (forward, at
  what it sees) with chance b, its balance; otherwise towards V (back off and turn). Born b = 0.9: a new-born
  creature is drawn to anything it sees. V also fires now and then by itself: a tumble, so the creature searches.
  The trace. A kicked cell keeps how many updates since its kick. Every cell keeps the news of each sensor: how old
  the freshest kick of R it has heard of is (R itself 0 when kicked; otherwise one more than the smaller of its own
  last news and its neighbours' smallest, from the mean of exp(-5 news)), and likewise of B. Without a new kick, no
  cell's news ever gets fresher (its neighbours' only grow older), whatever the order cells update in. So when R's
  news of B gets fresher, B has been kicked again; if that kick is younger than R's own, R forgets its own (and
  likewise B). So of R and B only the more recently kicked keeps a trace.
  A taste (+1 good, -1 bad, delivered to R and B) moves b by taste x exp(-trace / TRACE), towards 0 or 1: a
  three-factor rule. A bad taste also makes R and B recoil, sending pulses towards V, for a while.
  Every number is a whole number, re-rounded at each update (b in twentieths), so the noise can't move it.
"""

import numpy as np
from numba import njit

from rules import organs, wiring

FAR = organs.FAR
STEPS = 20.0        # b is kept in twentieths
INNATE = 18.0       # born b = 0.9
TRACE = 30.0        # how long a trace lasts (updates)
NEWS_MAX = 120.0    # news older than this is no news (5 x 120 < 700, so exp(-5 x news) is still a number)
N_BETA = 5.0
DISGUST = 40.0      # how long a bad taste lasts (updates) ...
RECOIL = 0.2        # ... sending a pulse towards V with this chance at each update
TUMBLE = 0.02       # V fires by itself with this chance at each update

VARS = wiring.VARS + ("balance", "kicked", "disgust", "news_r", "news_b")
SHARED = wiring.SHARED + (("news_r", N_BETA), ("news_b", N_BETA))
INPUTS = ("see", "taste", "learn")          # learn: the experiment's control (with it off, tastes don't move b)
OUTPUTS = wiring.OUTPUTS
BIRTH = {**wiring.BIRTH, "balance": INNATE, "kicked": FAR, "disgust": 0.0, "news_r": NEWS_MAX, "news_b": NEWS_MAX}
BIRTH_NOISE = wiring.BIRTH_NOISE
BALANCE, KICKED, DISGUSTED, NEWS = range(len(wiring.VARS), len(wiring.VARS) + 4)     # NEWS + 0: of R, + 1: of B
M_NEWS = len(wiring.SHARED)
SEE, TASTE, LEARN = range(3)
V_FIRED = wiring.BASE + 7 + wiring.FIRED


@njit
def update(me, mean, count, inp, out):
    organs.update(me, mean, count, inp, out)
    name = np.round(out[organs.NAME])
    source = name == wiring.SOURCES[0] or name == wiring.SOURCES[1]
    b = min(max(np.round(me[BALANCE]), 0.0), STEPS)
    kicked = min(np.round(me[KICKED]) + 1.0, FAR)
    for k in range(2):                                      # news of R, of B
        heard = np.round(organs._smallest(count * mean[M_NEWS + k], N_BETA)) + 1.0
        if name != wiring.SOURCES[k] and heard < np.round(me[NEWS + k]) and heard < kicked:
            kicked = FAR                                    # the other was kicked after me: forget my kick
        out[NEWS + k] = min(np.round(me[NEWS + k]) + 1.0, heard, NEWS_MAX)
    send_a = send_v = False
    if source and inp[SEE] > 0.5:
        kicked = 0.0
        out[NEWS + (0 if name == wiring.SOURCES[0] else 1)] = 0.0
        if np.random.random() < b / STEPS:
            send_a = True
        else:
            send_v = True
    disgust = max(np.round(me[DISGUSTED]) - 1.0, 0.0)
    taste = inp[TASTE]
    if source and taste != 0.0:
        if inp[LEARN] > 0.5:
            trace = np.exp(-kicked / TRACE)
            b = np.round(b + taste * trace * (b if taste < 0.0 else STEPS - b))
        if taste < 0.0:
            disgust = DISGUST
    if source and disgust > 0.0 and np.random.random() < RECOIL:
        send_v = True
    out[BALANCE], out[KICKED], out[DISGUSTED] = b, kicked, disgust
    wiring.stage(me, mean, count, out, 0, send_a)
    wiring.stage(me, mean, count, out, 1, send_v)
    if name == wiring.SINKS[1] and np.random.random() < TUMBLE:
        out[V_FIRED] += 1.0                                 # a tumble
