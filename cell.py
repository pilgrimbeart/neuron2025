"""How a cell behaves: the whole of it. Every cell runs the same rule, and the rule is what evolution finds.

Each cell holds a few numbers, VARIABLES, each kept within -1..1: v (fast; sensor pulses kick it, and an actuator
fires when it rises through FIRE_LEVEL), w (fast, free for evolution to use, e.g. as recovery), m (slow, free to use
as memory) and t (fast; tastes kick it, up for food and down for poison, like a neuromodulator rather than a spike).

Every tick, every cell looks at its own variables and at each variable of each of its 8 neighbours separately, in
compass order (DIRECTIONS); a missing neighbour (at the edge, or not yet born) reads as 0. Each variable x changes by

    x += RATE[x] x dt x (sum of TERMS weighted by the rule's coefficients for x)

where TERMS are 1, the inputs, and every product of two inputs (squares included). A rule is those coefficients;
most should be zero, so that the rule can be said in a few sentences (`describe`). Small rates make the rule an
equation of motion (like FitzHugh-Nagumo's): a cell can't flip every tick, so firing has to be a real excursion and
recovery; m's rate is ten times slower still. dt is the simulated time per tick; a rule that is a genuine
continuous-time system behaves much the same at any small dt.

A cell knows nothing of where it is: only its own variables and its neighbours'. The sheet is a list of cells, each
with its neighbour in each direction (`Sheet`); the body decides who neighbours whom.

Everything outside the cell is a kick: a sensor pulse to v, a taste to t.
"""

from __future__ import annotations

import numpy as np
from numba import njit

VARIABLES = ("v", "w", "m", "t")
RATES = np.array([0.1, 0.1, 0.01, 0.1])
DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
OFFSETS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))   # (dx, dy); y grows downwards
K = len(VARIABLES)
INPUTS = VARIABLES + tuple(f"{x}_{d}" for d in DIRECTIONS for x in VARIABLES)
N_INPUTS = len(INPUTS)                                        # 36
TERMS = ("1",) + INPUTS + tuple(f"{a}*{b}" if a != b else f"{a}^2"
                                for i, a in enumerate(INPUTS) for b in INPUTS[i:])
N_TERMS = len(TERMS)                                          # 703
N_PARAMS = K * N_TERMS                                        # the coefficients for each variable in turn
FIRE_LEVEL = 0.5
assert K == 4, "step's product loop is written out for 4 variables"


class Sheet:
    """Who neighbours whom: for n cells, neighbours[i, d] is cell i's neighbour in direction d, or n (a slot that is
    always 0) if it has none. A state is (K, n + 1)."""

    def __init__(self, neighbours: list[list[int | None]]):
        n = len(neighbours)
        self.n = n
        self.neighbours = np.array([[n if j is None else j for j in ns] for ns in neighbours], dtype=np.int64)


@njit(cache=True, fastmath=True)
def step(state, out, x, neighbours, alive, rule, dt):
    """One tick of dt for every live cell: reads state, writes out (both (K, n + 1)); x is scratch, (N_INPUTS, n).
    A cell that isn't alive stays at 0."""
    k = state.shape[0]
    n = neighbours.shape[0]
    for i in range(n):                                  # the inputs: own variables, then each neighbour's
        for a in range(k):
            x[a, i] = state[a, i]
        for d in range(neighbours.shape[1]):
            j = neighbours[i, d]
            for a in range(k):
                x[k + d * k + a, i] = state[a, j]
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
            c0, c1, c2, c3 = rule[t], rule[N_TERMS + t], rule[2 * N_TERMS + t], rule[3 * N_TERMS + t]
            for i in range(n):
                prod = x[p, i] * x[q, i]
                out[0, i] += c0 * prod
                out[1, i] += c1 * prod
                out[2, i] += c2 * prod
                out[3, i] += c3 * prod
            t += 1
    for a in range(k):                                  # move each variable by its rate, within -1..1
        r = RATES[a] * dt
        for i in range(n):
            out[a, i] = min(1.0, max(-1.0, state[a, i] + r * out[a, i])) if alive[i] else 0.0
        out[a, n] = 0.0


def describe(rule, threshold: float = 1e-3) -> str:
    """The rule in words: its non-zero terms."""
    lines = []
    for a, name in enumerate(VARIABLES):
        coefficients = rule[a * N_TERMS:(a + 1) * N_TERMS]
        parts = [f"{c:+.3f} {t}" if t != "1" else f"{c:+.3f}" for c, t in zip(coefficients, TERMS) if abs(c) > threshold]
        lines.append(f"d{name} = " + (" ".join(parts) if parts else "0"))
    return "\n".join(lines)


def translate(rule: np.ndarray, variables: tuple[str, ...]) -> np.ndarray:
    """A rule written in the earlier form (variables, and the mean of each over the neighbours, nx) as a rule for this
    one: nx becomes the sum of x over the 8 directions / 8 (exact for a cell with all 8 neighbours), and each product
    of old inputs expands into products of new ones. Variables the old rule lacks get zero coefficients."""
    rule = np.asarray(rule, dtype=float)
    if len(rule) == N_PARAMS:
        return rule
    old_inputs = variables + tuple("n" + x for x in variables)
    old_terms = [()] + [(a,) for a in old_inputs] + [(a, b) for i, a in enumerate(old_inputs) for b in old_inputs[i:]]
    index = {name: i for i, name in enumerate(INPUTS)}
    expand = lambda name: ({index[name]: 1.0} if name in index else
                           {index[f"{name[1:]}_{d}"]: 1 / len(DIRECTIONS) for d in DIRECTIONS})
    pair = {}
    for p in range(N_INPUTS):
        for q in range(p, N_INPUTS):
            pair[(p, q)] = len(pair) + 1 + N_INPUTS
    new = np.zeros(N_PARAMS)
    for a, name in enumerate(variables):
        base = VARIABLES.index(name) * N_TERMS
        for c, term in enumerate(old_terms):
            coefficient = rule[a * len(old_terms) + c]
            if len(term) == 0:
                new[base] += coefficient
            elif len(term) == 1:
                for p, wp in expand(term[0]).items():
                    new[base + 1 + p] += coefficient * wp
            else:
                for p, wp in expand(term[0]).items():
                    for q, wq in expand(term[1]).items():
                        new[base + pair[(min(p, q), max(p, q))]] += coefficient * wp * wq
    return new
