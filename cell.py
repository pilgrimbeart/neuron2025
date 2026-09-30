"""How a cell behaves: the whole of it. Every cell runs the same rule, and the rule is what evolution finds.

Each cell holds a few numbers, VARIABLES, each kept within -1..1: v (fast; an actuator fires when it rises through
FIRE_LEVEL), w (fast, free for evolution to use, e.g. as recovery) and m (slow, free to use as memory). Every tick,
every cell looks at its own variables and the mean of each over its (up to) 8 neighbours (nv, nw, nm), and each
variable x changes by

    x += RATE[x] x (sum of TERMS weighted by the rule's coefficients for x)

where TERMS are 1, the inputs, and their pairwise products and squares. A rule is those coefficients; most should be
zero, so that the rule can be said in a sentence or two (`describe`). Small rates make the rule an equation of
motion (like FitzHugh-Nagumo's): a cell can't flip every tick, so firing has to be a real excursion and recovery;
m's rate is ten times slower still.

Everything outside the cell is a kick to v: a sensor pulse, or a taste.
"""

from __future__ import annotations

import numpy as np
from numba import njit

VARIABLES = ("v", "w", "m")
RATES = np.array([0.1, 0.1, 0.01])
K = len(VARIABLES)
INPUTS = VARIABLES + tuple("n" + x for x in VARIABLES)
N_TERMS = 1 + 2 * K + 2 * K * (2 * K + 1) // 2
N_PARAMS = K * N_TERMS           # the coefficients for v, then w, then m
FIRE_LEVEL = 0.5


def terms(variables: tuple[str, ...]) -> tuple[str, ...]:
    inputs = variables + tuple("n" + x for x in variables)
    return ("1",) + inputs + tuple(f"{a}*{b}" if a != b else f"{a}^2"
                                   for i, a in enumerate(inputs) for b in inputs[i:])


TERMS = terms(VARIABLES)


@njit(cache=True)
def step(state, alive, rule):
    """One tick for every cell. state: (K, n, n) in -1..1; alive: bool (n, n); rule: N_PARAMS coefficients.
    Returns the new state."""
    k, n0, n1 = state.shape
    out = state.copy()
    x = np.empty(2 * k)
    t = np.empty(N_TERMS)
    for i in range(n0):
        for j in range(n1):
            if not alive[i, j]:
                continue
            for a in range(k):
                x[a] = state[a, i, j]
                x[k + a] = 0.0
            count = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    p, q = i + di, j + dj
                    if 0 <= p < n0 and 0 <= q < n1 and alive[p, q]:
                        for a in range(k):
                            x[k + a] += state[a, p, q]
                        count += 1
            if count:
                for a in range(k):
                    x[k + a] /= count
            t[0] = 1.0
            for a in range(2 * k):
                t[1 + a] = x[a]
            c = 1 + 2 * k
            for a in range(2 * k):
                for b in range(a, 2 * k):
                    t[c] = x[a] * x[b]
                    c += 1
            for a in range(k):
                d = 0.0
                for c in range(N_TERMS):
                    d += rule[a * N_TERMS + c] * t[c]
                out[a, i, j] = min(1.0, max(-1.0, state[a, i, j] + RATES[a] * d))
    return out


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
