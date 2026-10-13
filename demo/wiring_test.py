"""Measures the wiring rule on the 49 sheet: send pulses from R and B down each channel; how many arrive, how soon.

    python wiring_test.py [LIVES] [WARMUP]
"""
import sys, time, numpy as np
import sheet
from rules import wiring

N = 49
NAMES = {1: "R", 2: "B", 3: "A", 4: "V"}


def life(seed, warmup, trials=40, wait=300):
    s = sheet.Sheet(wiring, N, seed)
    s.step(warmup)
    paths = [int((s.view(f"path_{c}") > 0.5).sum()) for c in "av"]
    got, late = {}, {}
    rng = np.random.default_rng(seed)
    for i in range(trials):
        src, ch = 1 + i % 2, (i // 2) % 2
        sink, c = 3 + ch, "av"[ch]
        s.take(s.cells("name", sink), f"fired_{c}")
        s.give(s.cells("name", src), f"send_{c}")
        for t in range(wait):
            s.step(1)
            if s.take(s.cells("name", sink), f"fired_{c}").sum() > 0:
                late.setdefault((src, ch), []).append(t)
                break
        got[(src, ch)] = got.get((src, ch), 0) + (t < wait - 1)
        s.step(int(rng.integers(20, 40)))
    return paths, got, late, trials // 4


if __name__ == "__main__":
    lives = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    for seed in range(lives):
        t0 = time.time()
        paths, got, late, each = life(seed, warmup)
        parts = [f"{NAMES[src]}->{'AV'[ch]} {got.get((src, ch), 0)}/{each} in {np.mean(late.get((src, ch), [np.nan])):.0f}"
                 for src in (1, 2) for ch in (0, 1)]
        print(f"seed {seed}: path cells to A {paths[0]}, to V {paths[1]};  arrived, mean ticks: {'  '.join(parts)}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
