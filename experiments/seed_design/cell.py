"""How a cell behaves: the whole of it. Every cell runs the same rule, and the rule is what evolution finds.

Each cell holds a few numbers, VARIABLES, each kept within -1..1: v (fast; sensor pulses kick it, and an actuator
fires when it rises through FIRE_LEVEL), w (fast, free for evolution to use, e.g. as recovery), m (slow, free to use
as memory) and t (fast; tastes kick it, up for food and down for poison, like a neuromodulator rather than a spike).
Every tick, every cell looks at its own variables and the mean of each over its neighbours (nv, nw, nm, nt), and each
variable x changes by

    x += RATE[x] x dt x (sum of TERMS weighted by the rule's coefficients for x)

where TERMS are 1, the inputs, and their pairwise products and squares. A rule is those coefficients; most should be
zero, so that the rule can be said in a sentence or two (`describe`). Small rates make the rule an equation of
motion (like FitzHugh-Nagumo's): a cell can't flip every tick, so firing has to be a real excursion and recovery;
m's rate is ten times slower still. dt is the simulated time per tick (1 = the tick the rules were evolved at); a
rule that is a genuine continuous-time system behaves much the same at half or a quarter of it.

A cell knows nothing of where it is: only its own variables and its neighbours'. The sheet is a list of cells, each
with up to 8 neighbours (`Sheet`); the body decides who neighbours whom. Many lives of one sheet are stepped together
(the same rule, each cell side by side with its counterparts in the other lives), which is only a matter of speed.

Everything outside the cell is a kick: a sensor pulse to v, a taste to t.
"""

from __future__ import annotations

import numpy as np
from numba import njit

VARIABLES = ("v", "w", "m", "t")
RATES = np.array([0.1, 0.1, 0.01, 0.1], dtype=np.float32)
DTYPE = np.float32               # a cell's numbers
K = len(VARIABLES)
N_INPUTS = 2 * K
N_TERMS = 1 + N_INPUTS + N_INPUTS * (N_INPUTS + 1) // 2
N_PARAMS = K * N_TERMS           # the coefficients for each variable in turn
FIRE_LEVEL = 0.5
MAX_NEIGHBOURS = 8


def terms(variables: tuple[str, ...]) -> tuple[str, ...]:
    inputs = variables + tuple("n" + x for x in variables)
    return ("1",) + inputs + tuple(f"{a}*{b}" if a != b else f"{a}^2"
                                   for i, a in enumerate(inputs) for b in inputs[i:])


TERMS = terms(VARIABLES)


class Sheet:
    """Who neighbours whom, for n cells: each cell's up to 8 neighbours, the rest pointing at cell n, a slot that is
    always zero; and 1 / (number of neighbours), to take the mean."""

    def __init__(self, neighbours: list[list[int]]):
        n = len(neighbours)
        self.n = n
        self.neighbours = np.full((n, MAX_NEIGHBOURS), n, dtype=np.int64)
        self.weight = np.zeros(n, dtype=DTYPE)
        for i, ns in enumerate(neighbours):
            self.neighbours[i, :len(ns)] = ns
            self.weight[i] = 1.0 / len(ns) if ns else 0.0


def _source() -> str:
    """The source of `_change`: the rule written out as one expression per variable, term by term, inside a loop
    over lives, so that the compiler keeps a life's inputs and the coefficients in registers and works on 8 lives at
    once. It is the same sum as the TERMS description above, only spelled out."""
    inputs = [f"i{p}" for p in range(N_INPUTS)]
    lines = ["def _change(x, rule, change, lives):"]
    lines += [f"    c{t} = rule[{t}]" for t in range(N_PARAMS)]
    lines += ["    for l in range(lives):"]
    lines += [f"        {name} = x[{p}, l]" for p, name in enumerate(inputs)]
    for a in range(K):
        sum_ = [f"c{a * N_TERMS}"] + [f"c{a * N_TERMS + 1 + p} * {name}" for p, name in enumerate(inputs)]
        t = 1 + N_INPUTS
        for p in range(N_INPUTS):
            for q in range(p, N_INPUTS):
                sum_.append(f"c{a * N_TERMS + t} * {inputs[p]} * {inputs[q]}")
                t += 1
        lines.append(f"        d{a} = " + " + ".join(sum_))
    lines += [f"        change[{a}, l] = d{a}" for a in range(K)]
    return "\n".join(lines)


_namespace = {}
exec(_source(), _namespace)
_change = njit(fastmath=True)(_namespace["_change"])


@njit(fastmath=True)
def step(state, out, x, neighbours, weight, rule, dt):
    """One tick for every cell of L lives. state and out are (K, (n + 1) * L): for each variable, the cells in
    order, each cell's L lives side by side, then the zero slot's. x is scratch, (N_INPUTS + K, L). Cell by cell:
    gather a cell's inputs over its lives, work out every variable's change (`_change`), and move it."""
    k = state.shape[0]
    n = neighbours.shape[0]
    lives = state.shape[1] // (n + 1)
    change = x[N_INPUTS:]                               # rows of x after the inputs: the changes
    one = np.float32(1.0)
    for i in range(n):
        base = i * lives
        for a in range(k):                              # the inputs: own variables, and neighbours' means
            for l in range(lives):
                x[a, l] = state[a, base + l]
                x[k + a, l] = 0.0
            for j in range(neighbours.shape[1]):
                other = neighbours[i, j] * lives
                for l in range(lives):
                    x[k + a, l] += state[a, other + l]
            for l in range(lives):
                x[k + a, l] *= weight[i]
        _change(x, rule, change, lives)
        for a in range(k):                              # move each variable by its rate, within -1..1
            r = RATES[a] * np.float32(dt)
            for l in range(lives):
                out[a, base + l] = min(one, max(-one, state[a, base + l] + r * change[a, l]))
    for a in range(k):
        for c in range(n * lives, (n + 1) * lives):
            out[a, c] = 0.0


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
