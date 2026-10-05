"""How a cell behaves: the whole of it. Every cell runs the same rule, and the rule is what evolution finds.

Each cell holds a few numbers, VARIABLES, each kept within -1..1: v (fast; sensor pulses kick it, and an actuator
fires when it rises through FIRE_LEVEL), w (fast, free for evolution to use, e.g. as recovery), m (slow, free to use
as memory), t (fast; tastes kick it, up for food and down for poison, like a neuromodulator rather than a spike), and
e (fast) and z (slow), free for learning to use, e.g. as an eligibility trace and a gate on where learning happens.
Every tick, every cell looks at its own variables and the mean of each over its neighbours (nv, nw, ...), and each
variable x changes by

    x += RATE[x] x dt x (sum of TERMS weighted by the rule's coefficients for x)

where TERMS are 1, the inputs, their pairwise products and squares, and the cubic products of a cell's own variables
(such as e*t*z: a three-factor learning rule is one term). The quadratic terms come first, so a rule from before the
cubic terms is the start of one with them. A rule is those coefficients; most should be
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

VARIABLES = ("v", "w", "m", "t", "e", "z")
RATES = np.array([0.1, 0.1, 0.01, 0.1, 0.1, 0.01], dtype=np.float32)
DTYPE = np.float32               # a cell's numbers
K = len(VARIABLES)
N_INPUTS = 2 * K
FIRE_LEVEL = 0.5
MAX_NEIGHBOURS = 8


def factors(k: int) -> list[tuple[int, ...]]:
    """Each term as the inputs it multiplies (inputs 0..k-1 are a cell's own variables, k..2k-1 its neighbours'
    means): 1, each input, each pair, then each triple of own variables."""
    n = 2 * k
    return ([()] + [(p,) for p in range(n)] + [(p, q) for p in range(n) for q in range(p, n)]
            + [(p, q, r) for p in range(k) for q in range(p, k) for r in range(q, k)])


def terms(variables: tuple[str, ...]) -> tuple[str, ...]:
    """The terms' names: "1", "v", "nv*w", "v^2", "e*t*z", "v^2*w", ..."""
    inputs = variables + tuple("n" + x for x in variables)
    named = []
    for f in factors(len(variables)):
        powers = [inputs[p] + (f"^{f.count(p)}" if f.count(p) > 1 else "") for p in sorted(set(f))]
        named.append("*".join(powers) or "1")
    return tuple(named)


TERMS = terms(VARIABLES)
N_TERMS = len(TERMS)
N_PARAMS = K * N_TERMS           # the coefficients for each variable in turn


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
    """The source of `_change`: the rule written out term by term, one loop over lives per variable (small enough
    for the compiler to turn into vector instructions, 8 lives at once). It is the same sum as the TERMS description
    above, only spelled out."""
    lines = ["def _change(x, rule, change, lives):"]
    for a in range(K):
        lines += [f"    c{t} = rule[{a * N_TERMS + t}]" for t in range(N_TERMS)]
        sum_ = [f"c{t}" + "".join(f" * x[{p}, l]" for p in f) for t, f in enumerate(factors(K))]
        lines += ["    for l in range(lives):", f"        change[{a}, l] = " + " + ".join(sum_)]
    return "\n".join(lines) + "\n"


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
    """A rule written for fewer variables or terms, as a rule for these: the same coefficients, the new terms zero.
    A rule's terms are a start of terms(variables) (one from before the cubic terms has only the quadratic ones)."""
    if len(rule) == N_PARAMS:
        return np.asarray(rule, dtype=float)
    per = len(rule) // len(variables)
    old_terms = terms(variables)[:per]
    new = np.zeros(N_PARAMS)
    for a, name in enumerate(variables):
        for c, term in enumerate(old_terms):
            new[VARIABLES.index(name) * N_TERMS + TERMS.index(term)] = rule[a * per + c]
    return new
