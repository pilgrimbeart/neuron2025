"""Layouts for every bundled pattern, generated from code so they can be rebuilt and re-verified.

    python gates.py            rewrite every pattern file in patterns/
    python gates.py xor osc    rewrite just those

All patterns use physics.DEFAULT_PARAMS. verify.py checks the saved files against these layouts and
against each pattern's expected behaviour.
"""

from __future__ import annotations

import sys

import physics
from model import SimulationConfig, State
from persistence import save_snapshot

Cells = set[tuple[int, int]]


def hline(y: int, x0: int, x1: int) -> Cells:
    return {(x, y) for x in range(min(x0, x1), max(x0, x1) + 1)}


def vline(x: int, y0: int, y1: int) -> Cells:
    return {(x, y) for y in range(min(y0, y1), max(y0, y1) + 1)}


def diode(target: tuple[int, int], direction: tuple[int, int]) -> Cells:
    """The fork of a one-way gate into target, for pulses travelling in direction (dx, dy).

    The line arriving at target - 2*direction splits into two arms that touch target only diagonally,
    leaving the gap cell target - direction empty. Two diagonal neighbours ignite target; target alone
    can't ignite either arm, so nothing comes back.
    """
    (tx, ty), (dx, dy) = target, direction
    px, py = -dy, dx
    return {(tx - k * dx + s * px, ty - k * dy + s * py) for k in (1, 2) for s in (1, -1)} | {(tx - 2 * dx, ty - 2 * dy)}


def transpose(cells: Cells) -> Cells:
    return {(y, x) for x, y in cells}


def mirror_x(cells: Cells, size: int) -> Cells:
    return {(size - 1 - x, y) for x, y in cells}


# --- simple patterns --------------------------------------------------------------------------------

def line():
    return hline(16, 0, 31), {'in': (0, 16), 'out': (31, 16)}, 32


def oneway():
    c = hline(16, 2, 12) | hline(16, 14, 29) | diode((14, 16), (1, 0))
    return c, {'a': (2, 16), 'b': (29, 16)}, 32


def and_gate():
    """Coincidence: the ends of a and b are both diagonal to the junction (16,16)."""
    c = hline(15, 2, 15) | hline(17, 2, 15) | hline(16, 16, 29)
    return c, {'a': (2, 15), 'b': (2, 17), 'out': (16, 16), 'edge': (29, 16)}, 32


def or_gate():
    """Two diodes feeding a shared relay; the output leaves from the relay's middle."""
    c = hline(16, 2, 10) | hline(16, 12, 19) | hline(16, 21, 29) | diode((12, 16), (1, 0)) | diode((19, 16), (-1, 0))
    c |= vline(15, 17, 28)
    return c, {'a': (2, 16), 'b': (29, 16), 'out': (15, 16), 'edge': (15, 28)}, 32


def osc():
    """A 60-cell ring with a diode in it, so a pulse can only circulate clockwise, forever.

    start joins just after the diode, so the half of the launch pulse that heads backwards dies at the
    diode's target at once. start and out each have their own diode, so the ring can't leak into start
    and nothing arriving at out can get in.
    """
    c = hline(8, 8, 14) | hline(8, 16, 23) | vline(23, 8, 23) | hline(23, 8, 23) | vline(8, 8, 23)
    c |= diode((16, 8), (1, 0))
    c |= vline(19, 0, 3) | diode((19, 5), (0, 1)) | vline(19, 5, 7)           # start, joining the ring at (19,8)
    c |= hline(16, 24, 25) | diode((27, 16), (1, 0)) | hline(16, 27, 31)      # out, leaving the ring at (23,16)
    return c, {'start': (19, 0), 'out': (31, 16)}, 32


# --- inhibit units: a long-armed diode whose inner arm a veto can pre-burn -------------------------------

def inhibit():
    """in passes to out unless inh arrives at the same time; inh alone never produces output.

    in goes through a diode (so inh's backward wave can't leave through in), a short bump (delay), then
    splits into two arms that meet diagonally at the target (24,16). inh arrives from the top through its
    own diode and pre-burns the inner arm near the target, so in's pulse arrives to find it refractory.
    """
    c = hline(16, 0, 3) | diode((5, 16), (1, 0)) | hline(16, 5, 8) | vline(8, 16, 19) | hline(19, 8, 12)
    c |= vline(12, 16, 19) | hline(16, 12, 15)
    c |= vline(15, 13, 20) | hline(13, 15, 23) | vline(23, 13, 15) | hline(20, 15, 23) | vline(23, 17, 20)
    c |= hline(16, 24, 31)
    c |= vline(21, 0, 9) | diode((21, 11), (0, 1)) | vline(21, 11, 12)
    return c, {'in': (0, 16), 'inh': (21, 0), 'out': (31, 16)}, 32


def xor():
    """out = (a inhibited by AND) OR (b inhibited by AND), with a, b and out all on the grid edge."""
    sy, feed, veto_y, outer = 11, 26, 7, 2
    c = vline(6, 4, 5)                                                                    # T_a (6,5) under the relay
    c |= hline(6, outer, 5) | vline(outer, 6, sy) | hline(6, 7, 9) | vline(9, 6, sy) | hline(sy, outer, 9)
    c |= {(10, veto_y), (11, veto_y)} | diode((11, veto_y), (-1, 0)) | hline(veto_y, 13, 15)
    c |= hline(feed, 0, 14)                                                               # a's feed; P_a = (12,feed)
    top, bottom = sy + 3, feed - 3
    c |= vline(12, top, feed - 1) | hline(top, 9, 12) | vline(9, top, bottom) | hline(bottom, 6, 9) | vline(6, sy + 1, bottom)
    c |= mirror_x(c, 32)
    c |= hline(3, 6, 25) | vline(15, 0, 2)                                                # OR relay and output
    c |= vline(15, veto_y + 1, feed - 2) | hline(veto_y, 15, 18)                          # veto trunk and fan-out
    c |= {(16, feed), (15, feed - 1)}                                                     # AND junction (15,feed-1)
    return c, {'a': (0, feed), 'b': (31, feed), 'and': (15, feed - 1), 'out': (15, 2), 'edge': (15, 0)}, 32


def cross():
    """a (left) -> a_out (right) and b (top) -> b_out (bottom); simultaneous pulses leave at both."""
    n, ta, relay, span = 48, 16, 10, 9
    cx = ta + relay
    join = ta + 2
    # b's half; a's half is its transpose
    c = vline(cx, 0, 5) | hline(5, cx - 4, cx) | vline(cx - 4, 5, 9) | hline(9, cx - 4, cx) | vline(cx, 9, ta - 2)
    c |= diode((cx, ta), (0, 1)) | vline(cx, ta, cx)
    sx = cx + 6
    tx = sx + span
    c |= hline(join, cx, sx) | vline(sx, join - 3, join + 4)
    c |= hline(join - 3, sx, tx - 1) | vline(tx - 1, join - 3, join - 1)
    c |= hline(join + 4, sx, tx - 1) | vline(tx - 1, join + 1, join + 4)
    c |= hline(join, tx, n - 1)
    vx = tx - 3
    c |= hline(2, cx, vx) | vline(vx, 2, join - 7) | diode((vx, join - 5), (0, 1)) | vline(vx, join - 5, join - 4)
    c |= transpose(c)
    # both-detector J, fed diagonally by a spur from each leg; its output reaches both outputs via diodes
    jx, jy = cx + 2, cx + 4
    out = tx + 3
    by = jy + 3
    c |= hline(jy - 1, join, jx - 1) | vline(jx + 1, join, jy - 1) | {(jx, jy)}
    c |= vline(jx, jy, out) | hline(out, join, jx) | hline(by, jx, out) | vline(out, join, by)
    c -= {(out - 4, by), (jx, out - 4)}
    c |= diode((out - 3, by), (1, 0)) | diode((jx, out - 3), (0, 1))
    return c, {'a': (0, cx), 'b': (cx, 0), 'both': (jx, jy), 'a_out': (n - 1, join), 'b_out': (join, n - 1)}, n


PATTERNS = {
    'line': line,
    'oneway': oneway,
    'and': and_gate,
    'or': or_gate,
    'xor': xor,
    'cross': cross,
    'osc': osc,
    'inhibit': inhibit,
}


def build_state(name: str) -> tuple[State, list[dict]]:
    cells, probes, size, *rest = PATTERNS[name]()
    teachers = rest[0] if rest else set()
    state = State((size, size))
    for xy in cells:
        state.set_kind(xy, physics.NORMAL)
    for xy in teachers:
        state.set_kind(xy, physics.TEACHER)
    return state, [{'xy': xy, 'label': label} for label, xy in probes.items()]


def write(name: str) -> None:
    state, probes = build_state(name)
    save_snapshot(name, state, probes, SimulationConfig(dict(physics.DEFAULT_PARAMS)))
    print(f'wrote patterns/{name}.json')


if __name__ == '__main__':
    for name in sys.argv[1:] or PATTERNS:
        write(name)
