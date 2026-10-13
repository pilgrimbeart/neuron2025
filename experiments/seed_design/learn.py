"""Taste-gated cutting: does a bad taste just after red's pulse cut red's channel and spare blue's?"""
import numpy as np, seed, cell
from chan2 import channel_sheet

def learning_sheet(eta, spread=2.0, fade=3.0, seed_=0):
    s, terms = channel_sheet(seed_=seed_)
    terms["m"]["w*t"] = eta                          # recently fired (w) x taste (t): cut after bad, grow after good
    terms["t"] = {"nt": spread, "t": -fade}          # taste spreads a little, and fades
    state = s.a.copy()
    s2 = seed.Sheet(seed.rule(terms), seed=seed_)
    s2.a[:] = state                                  # carry on from the settled labyrinth
    return s2

def reach(s, cell_i, ticks=250):
    """Cells fired by one pulse at cell_i, then quiet again (the sheet is copied, so this doesn't change it)."""
    a, b = s.a.copy(), s.b.copy()
    s.kick(cell_i); fired = np.zeros(s.n, bool)
    for _ in range(ticks):
        s.step(); fired |= s.var("v") > 0.5
    s.a[:], s.b[:] = a, b
    return fired.sum()

if __name__ == "__main__":
    for eta in (20.0, 40.0, 80.0):
        for seed_ in range(3):
            s = learning_sheet(eta, seed_=seed_)
            g = s.g; red, blue, taste = g.inputs[0, 0], g.inputs[0, 1], g.taste[0]
            before = reach(s, red), reach(s, blue)
            s.kick(red); s.step(20)                  # red fires ...
            s.kick(taste, -1.0, var="t"); s.step(400)   # ... then a bad taste nearby; let it settle
            after = reach(s, red), reach(s, blue)
            print(f"eta {eta}, seed {seed_}: reach red {before[0]:4d} -> {after[0]:4d},  blue {before[1]:4d} -> {after[1]:4d}")
