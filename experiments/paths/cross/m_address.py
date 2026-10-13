"""Mechanism 5: addresses. Four landmark cells on the rim (N, E, S, W) each send waves (breadcrumb-rule channels), so
every cell knows its count to each: coordinates u = dE - dW, v = dS - dN. A pulse carries an address, its input's
coordinates reflected through the centre (-u, -v), and at each step hops to the neighbour whose coordinates are
nearest the address; once its own coordinates are within NEAR of the address, it follows breadcrumbs (a thruster
channel) to the nearest thruster. No wiring; pulses cross freely; closed-loop."""
import sys, numpy as np
sys.path.insert(0, "/home/pilgrim/git/neuron2025")
import paths as P
import lab

def channel(seed, sinks, time=200):
    lives = P.life(seed, 1.0, P.Ends(sources=(), sinks=tuple(sinks)))
    for t, p, now in lives:
        if t >= time: break
    f = lives.gi_frame.f_locals
    return f["s"][P.COUNT].copy(), f["s"][P.CRUMB].copy(), f["tag"]

class Address:
    def __init__(self, near=3.0, jitter=0.01):
        self.near, self.jitter = near, jitter
    def grow(self, seed=0):
        P.JITTER = self.jitter
        inputs, thrusters = lab.layout()
        poles = [lab.at(a, lab.R) for a in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)]     # E, S, W, N (angle 0 = +x)
        d = [channel(seed + 31 * i, [pole])[0] for i, pole in enumerate(poles)]
        count, crumb, tag = channel(seed + 7, thrusters)
        P.JITTER = 0.01
        u, v = d[0] - d[2], d[1] - d[3]
        return {"u": u, "v": v, "crumb": crumb, "tag": tag, "thrusters": thrusters, "inside": lab.disc()}
    def inject(self, state, cell):
        return [("addr", cell, (-state["u"][cell], -state["v"][cell]), 0)]
    def step(self, state, pulses):
        out, hits = [], []
        u, v = state["u"], state["v"]
        for kind, (y, x), (tu, tv), n in pulses:
            if kind == "addr":
                here = np.hypot(u[y, x] - tu, v[y, x] - tv)
                if here < self.near or n > 120:
                    kind = "crumb"
                else:
                    best, bd = None, here
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if (dy or dx) and 0 <= yy < lab.SIZE and 0 <= xx < lab.SIZE and state["inside"][yy, xx]:
                                dd = np.hypot(u[yy, xx] - tu, v[yy, xx] - tv)
                                if dd < bd:
                                    best, bd = (yy, xx), dd
                    if best is None:
                        kind = "crumb"                      # a local minimum: hand over to breadcrumbs
                    else:
                        if best in state["thrusters"]:
                            hits.append(state["thrusters"].index(best)); continue
                        out.append(("addr", best, (tu, tv), n + 1)); continue
            c, tag = state["crumb"][y, x], state["tag"]
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (dy or dx) and 0 <= yy < lab.SIZE and 0 <= xx < lab.SIZE and tag[yy, xx] == c:
                        nxt = (yy, xx)
            if nxt is None:
                continue
            if nxt in state["thrusters"]:
                hits.append(state["thrusters"].index(nxt))
            else:
                out.append(("crumb", nxt, (tu, tv), n + 1))
        return out, hits

if __name__ == "__main__":
    for jitter in (0.01, 0.1, 0.3):
        res = []
        for seed in range(4):
            mech = Address(jitter=jitter); st = mech.grow(seed)
            ok, wrong, lost, d = lab.pulse_test(mech, st, *lab.layout(), repeats=1)
            res.append((round(ok, 2), round(wrong, 2), round(lost, 2)))
        print(f"5: addresses, jitter {jitter}: (correct, crosstalk, lost) per seed {res}", flush=True)

class AddressPlateau(Address):
    """As Address, but a pulse may step to the best neighbour even if no closer (within SLACK), never straight back."""
    def __init__(self, near=3.0, jitter=0.01, slack=0.5):
        super().__init__(near, jitter); self.slack = slack
    def inject(self, state, cell):
        return [("addr", cell, (-state["u"][cell], -state["v"][cell]), 0, cell)]
    def step(self, state, pulses):
        out, hits = [], []
        u, v = state["u"], state["v"]
        for kind, (y, x), (tu, tv), n, prev in pulses:
            if kind == "addr":
                here = np.hypot(u[y, x] - tu, v[y, x] - tv)
                if here < self.near or n > 150:
                    kind = "crumb"
                else:
                    best, bd = None, here + self.slack
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if (dy or dx) and 0 <= yy < lab.SIZE and 0 <= xx < lab.SIZE and state["inside"][yy, xx] \
                                    and (yy, xx) != prev:
                                dd = np.hypot(u[yy, xx] - tu, v[yy, xx] - tv)
                                if dd < bd:
                                    best, bd = (yy, xx), dd
                    if best is None:
                        kind = "crumb"
                    else:
                        if best in state["thrusters"]:
                            hits.append(state["thrusters"].index(best)); continue
                        out.append(("addr", best, (tu, tv), n + 1, (y, x))); continue
            r, h = Address.step(self, state, [("crumb", (y, x), (tu, tv), n)])
            out += [p + ((y, x),) for p in r]; hits += h
        return out, hits
