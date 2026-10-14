"""Measures the organs rule: when the coordinates are exact and the names right (R, B and V one cell each, A the rest of
the tail's edge), and whether they stay so.

    python organs_test.py [LIVES] [SIZE]
"""
import sys
import time

import numpy as np

import sheet
from rules import organs


def right(s, n):
    """The coordinates are exact rows and columns, and the names are where they should be."""
    y, x = np.round(s.view("y")), np.round(s.view("x"))
    r = np.arange(n)
    lines = [np.broadcast_to(r[:, None], (n, n)), np.broadcast_to(n - 1 - r[:, None], (n, n)),
             np.broadcast_to(r[None, :], (n, n)), np.broadcast_to(n - 1 - r[None, :], (n, n))]
    if not (any((y == g).all() for g in lines) and any((x == g).all() for g in lines)):
        return False
    names = np.round(s.view("name"))
    one = all((names == k).sum() == 1 for k in (organs.R, organs.B, organs.V))
    return one and (names == organs.A).sum() == n - 1 and ((names == organs.V) == ((x == 0) & (y == 0))).all()


def life(n, seed, ticks):
    s = sheet.Sheet(organs, n, seed, update=0.9)
    settled = None
    for t in range(ticks):
        s.step(1)
        ok = right(s, n)
        settled = (settled if settled is not None else t) if ok else None
    return settled, int((s.view("origin") > 0.5).sum())


if __name__ == "__main__":
    lives = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 49
    for seed in range(lives):
        t0 = time.time()
        settled, origins = life(n, seed, 30 * n)
        print(f"n {n} seed {seed}: right for good from tick {settled} (of {30 * n}), origins {origins}"
              f"  ({time.time() - t0:.1f}s)", flush=True)
