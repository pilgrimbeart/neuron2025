"""Corner flags, carefully. Everyone except the corners averages its side neighbours (noise in every update). A
corner holds u = 0 until it raises its flag (u = 1, for good). It watches its neighbours' average, smoothed (rate
SMOOTH), against a baseline:
  - a step begins when the smoothed average leaves the baseline by more than TH;
  - it has settled when, at each doubling of the time since it began, the value changed by less than SETTLE of the
    step so far; its size is recorded (its own raise is not recorded), and the baseline moves to the new level;
  - it raises its flag only when no step is in progress, with a small chance per update: 1 / (WAIT x the time the
    last step took to settle), or 1 / (WAIT x its age) before it has seen any.
Labels from the order of events and which of the three recorded steps was the smallest (see label())."""
import sys, numpy as np
from sim import means, degree
from multiprocessing import Pool
SMOOTH, SETTLE, WAIT, TH, FLAG = 0.01, 0.05, 10.0, 0.01, 10.0
EXTRAPOLATE = False
class Corner:
    def __init__(self):
        self.raised = None; self.ms = None; self.base = None; self.start = None; self.check = None
        self.steps = []; self.settle = None; self.own = False; self.hist = []; self.est = None
    def see(self, a, m, rng):
        self.ms = m if self.ms is None else self.ms + SMOOTH * (m - self.ms)
        if self.base is None:
            self.base = self.ms
            return
        change = self.ms - self.base
        if self.start is None:
            if change > TH * FLAG:                                             # flags only go up
                self.start, self.check = a, (a, self.ms)
            elif self.raised is None:
                hazard = 1.0 / (WAIT * (self.settle if self.settle else max(a, 1 / SMOOTH)))
                if rng.random() < hazard:
                    self.raised = a; self.own = True
            return
        if a - self.start >= max(2 * (self.check[0] - self.start), 1 / SMOOTH):   # first check after the smoothing time, then at doublings
            if EXTRAPOLATE:                                                     # where the step is heading (Aitken), from
                self.hist = (self.hist + [self.ms])[-3:]                         # the last three values at doubling times
                if len(self.hist) == 3:
                    y1, y2, y3 = self.hist; d1, d2 = y2 - y1, y3 - y2
                    est = y3 + d2 * d2 / (d1 - d2) if abs(d1 - d2) > 1e-12 and d2 / d1 < 0.9 and d1 > 0 else None
                    if est is not None and self.est is not None and abs(est - self.est) < SETTLE * abs(est - self.base):
                        change = est - self.base; settled = True
                    else:
                        settled = False
                    self.est = est
                else:
                    settled = False
            else:
                settled = abs(self.ms - self.check[1]) < SETTLE * abs(change)
            if settled:                                                         # settled
                if not self.own:
                    self.steps.append((self.start, change))
                self.settle = a - self.start if self.settle is None else max(self.settle, a - self.start)
                self.base = self.ms; self.start = None; self.own = False; self.hist = []; self.est = None
            else:
                self.check = (a, self.ms)
    def label(self):
        if self.raised is None or len(self.steps) < 3: return None
        st = sorted(self.steps[:3])
        diag = min(range(3), key=lambda i: st[i][1])
        before = [i for i in range(3) if st[i][0] < self.raised]
        if not before: return (0, 0)
        if before[0] == diag: return (1, 1)
        return (0, 1) if diag in before else (1, 0)
def job(a):
    n, seed, sigma, maxticks = a
    rng = np.random.default_rng(seed); deg = degree(n)
    cs = [(0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 0)]
    u = np.zeros((n, n)); age = np.zeros((n, n)); C = [Corner() for _ in cs]
    for t in range(maxticks):
        go = rng.random((n, n)) < 0.5
        age = age + go
        m = means(u[..., None], deg)[..., 0]
        new = m + sigma * rng.standard_normal((n, n))
        for k, c in enumerate(cs):
            new[c] = FLAG if C[k].raised is not None else 0.0
            if go[c]: C[k].see(age[c], m[c], rng)
        u = np.where(go, new, u)
        if all(x.label() is not None for x in C): break
    labs = [x.label() for x in C]
    square = None not in labs and len(set(labs)) == 4 and all(abs(labs[k][0] - labs[(k + 1) % 4][0]) + abs(labs[k][1] - labs[(k + 1) % 4][1]) == 1 for k in range(4))
    return square, labs, t, [len(x.steps) for x in C], [round(min(s[1] for s in x.steps[:3]) / max(s[1] for s in x.steps[:3]), 2) if len(x.steps) >= 3 else None for x in C]
if __name__ == "__main__":
    if len(sys.argv) > 3: SETTLE, WAIT = float(sys.argv[3]), float(sys.argv[4])
    if len(sys.argv) > 5: EXTRAPOLATE = sys.argv[5] == 'aitken'
    for n in (int(x) for x in sys.argv[1].split(",")):
        with Pool(8) as pool:
            r = pool.map(job, [(n, s, 1e-2, int(sys.argv[2])) for s in range(8)])
        print(f"n {n} noise 1e-2: proper square of labels {sum(x[0] for x in r)}/8; e.g. labels {r[0][1]} after {r[0][2]} ticks, steps seen {r[0][3]}, smallest/largest step {r[0][4]}", flush=True)
        for x in r[1:]:
            if not x[0]: print("   failure:", x[1], "ticks", x[2], "steps", x[3], "ratio", x[4])
