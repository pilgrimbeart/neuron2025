"""Measures the organs rule: when each name is held by exactly one edge cell, and whether it stays so."""
import sys, time, numpy as np
import sheet
from rules import organs

def life(n, seed, ticks):
    s = sheet.Sheet(organs, n, seed)
    settled, last = None, None
    for t in range(ticks):
        s.step(1)
        names = s.view("name")
        ok = all((np.abs(names - k) < 0.5).sum() == 1 for k in range(1, 5))
        where = tuple(tuple(int(v) for v in np.argwhere(np.abs(names - k) < 0.5)[0]) for k in range(1, 5)) if ok else None
        if not ok or where != last:
            settled = None if not ok else t
        last = where
    origins = int((s.view("origin") > 0.5).sum())
    return settled, origins, last

if __name__ == "__main__":
    for n in (int(a) for a in sys.argv[1:] or (49, 97)):
        for seed in range(6):
            t0 = time.time()
            settled, origins, where = life(n, seed, 60 * n)
            print(f"n {n} seed {seed}: names settled for good at tick {settled} (of {60 * n}), origins {origins}, "
                  f"R B A V at {where}  ({time.time()-t0:.1f}s)", flush=True)
