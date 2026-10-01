"""How a cell behaves: the whole of it. Every cell runs the same rule, and the rule is what evolution finds.

Each cell holds a few numbers, VARIABLES, each kept within -1..1: v (fast; an actuator fires when it rises through
FIRE_LEVEL), w (fast, free for evolution to use, e.g. as recovery) and m (slow, free to use as memory). Every tick,
every cell looks at its own variables and the mean of each over its neighbours (nv, nw, nm), and each variable x
changes by

    x += RATE[x] x dt x (sum of TERMS weighted by the rule's coefficients for x)

where TERMS are 1, the inputs, and their pairwise products and squares. A rule is those coefficients; most should be
zero, so that the rule can be said in a sentence or two (`describe`). Small rates make the rule an equation of
motion (like FitzHugh-Nagumo's): a cell can't flip every tick, so firing has to be a real excursion and recovery;
m's rate is ten times slower still. dt is the simulated time per tick (1 = the tick the rules were evolved at); a
rule that is a genuine continuous-time system behaves much the same at half or double it.

A cell knows nothing of where it is: only its own variables and its neighbours'. The sheet is a list of cells, each
with up to 8 neighbours (`neighbours`); the body decides who neighbours whom.

Everything outside the cell is a kick to v: a sensor pulse, or a taste.
"""

from __future__ import annotations

import numpy as np
from numba import njit

VARIABLES = ("v", "w", "m")
RATES = np.array([0.1, 0.1, 0.01])
K = len(VARIABLES)
N_INPUTS = 2 * K
N_TERMS = 1 + N_INPUTS + N_INPUTS * (N_INPUTS + 1) // 2
N_PARAMS = K * N_TERMS           # the coefficients for v, then w, then m
FIRE_LEVEL = 0.5
MAX_NEIGHBOURS = 8


def terms(variables: tuple[str, ...]) -> tuple[str, ...]:
    inputs = variables + tuple("n" + x for x in variables)
    return ("1",) + inputs + tuple(f"{a}*{b}" if a != b else f"{a}^2"
                                   for i, a in enumerate(inputs) for b in inputs[i:])


TERMS = terms(VARIABLES)


class Sheet:
    """Who neighbours whom, and the working space for stepping a sheet of n cells. A state is (K, n + 1): the
    cells' variables, then a slot that is always zero, where a missing neighbour points."""

    def __init__(self, neighbours: list[list[int]]):
        n = len(neighbours)
        self.n = n
        self.neighbours = np.full((n, MAX_NEIGHBOURS), n, dtype=np.int64)
        self.weight = np.zeros(n)
        for i, ns in enumerate(neighbours):
            self.neighbours[i, :len(ns)] = ns
            self.weight[i] = 1.0 / len(ns) if ns else 0.0


@njit(cache=True, fastmath=True)
def step(state, out, x, neighbours, weight, rule, dt):
    """One tick of dt for every cell: reads state, writes out (both (K, n + 1)); x is scratch, (N_INPUTS, n)."""
    k = state.shape[0]
    n = neighbours.shape[0]
    for a in range(k):                                  # the inputs: own variables, and neighbours' means
        for i in range(n):
            x[a, i] = state[a, i]
            s = 0.0
            for j in range(neighbours.shape[1]):
                s += state[a, neighbours[i, j]]
            x[k + a, i] = s * weight[i]
    for a in range(k):                                  # constant and linear terms
        c0 = rule[a * N_TERMS]
        for i in range(n):
            out[a, i] = c0
        for p in range(N_INPUTS):
            c = rule[a * N_TERMS + 1 + p]
            for i in range(n):
                out[a, i] += c * x[p, i]
    t = 1 + N_INPUTS                                    # products and squares
    for p in range(N_INPUTS):
        for q in range(p, N_INPUTS):
            for a in range(k):
                c = rule[a * N_TERMS + t]
                for i in range(n):
                    out[a, i] += c * x[p, i] * x[q, i]
            t += 1
    for a in range(k):                                  # move each variable by its rate, within -1..1
        r = RATES[a] * dt
        for i in range(n):
            out[a, i] = min(1.0, max(-1.0, state[a, i] + r * out[a, i]))
        out[a, n] = 0.0


def describe(rule, threshold: float = 1e-3) -> str:
    """The rule in words: its non-zero terms."""
    lines = []
    for a, name in enumerate(VARIABLES):
        coefficients = rule[a * N_TERMS:(a + 1) * N_TERMS]
        parts = [f"{c:+.3f} {t}" if t != "1" else f"{c:+.3f}" for c, t in zip(coefficients, TERMS) if abs(c) > threshold]
        lines.append(f"d{name} = " + (" ".join(parts) if parts else "0"))
    return "\n".join(lines)


def widen(rule: np.ndarray, variables: tuple[str, ...]) -> np.ndarray:
    """A rule written for fewer variables, as a rule for these: the same coefficients, the new terms zero."""
    if len(rule) == N_PARAMS:
        return np.asarray(rule, dtype=float)
    old_terms = terms(variables)
    new = np.zeros(N_PARAMS)
    for a, name in enumerate(variables):
        for c, term in enumerate(old_terms):
            new[VARIABLES.index(name) * N_TERMS + TERMS.index(term)] = rule[a * len(old_terms) + c]
    return new
