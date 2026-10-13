"""Mechanism 3: rays. A pulse leaves its input heading inwards (up the distance-from-the-edge gradient, worked out
from the neighbours), flies straight (it carries its direction and where it is within the cell, as a ray of light
would), and once it reaches the far edge follows breadcrumbs (one shared channel, thrusters as sinks) to the nearest
thruster. WOBBLE is random error in the direction at each step."""
import sys, numpy as np
sys.path.insert(0, "/home/pilgrim/git/neuron2025")
import paths as P
import lab

class Rays:
    def __init__(self, wobble=0.0, edge_band=2.0):
        self.wobble, self.edge_band = wobble, edge_band
    def grow(self, seed=0):
        inputs, thrusters = lab.layout()
        lives = P.life(seed, 1.0, P.Ends(sources=(), sinks=tuple(thrusters)))
        for t, p, now in lives:
            if t >= 200: break
        f = lives.gi_frame.f_locals
        inside = lab.disc()
        # distance from the edge, as the cells would learn it from waves sent by the rim (here worked out directly)
        edge = np.full((lab.SIZE, lab.SIZE), 0.0)
        from scipy.ndimage import distance_transform_edt
        edge = distance_transform_edt(inside)
        self.rng = np.random.default_rng(seed)
        return {"crumb": f["s"][P.CRUMB].copy(), "tag": f["tag"], "edge": edge, "inside": inside, "thrusters": thrusters}
    def inject(self, state, cell):
        y, x = cell
        e = state["edge"]
        gy = e[min(y + 1, lab.SIZE - 1), x] - e[max(y - 1, 0), x]      # the gradient, from the neighbours
        gx = e[y, min(x + 1, lab.SIZE - 1)] - e[y, max(x - 1, 0)]
        norm = np.hypot(gy, gx) or 1.0
        return [("ray", float(y), float(x), gy / norm, gx / norm, 0)]
    def step(self, state, pulses):
        out, hits = [], []
        for pulse in pulses:
            if pulse[0] == "ray":
                _, y, x, dy, dx, n = pulse
                if self.wobble:
                    a = np.arctan2(dy, dx) + self.rng.normal(0, self.wobble)
                    dy, dx = np.sin(a), np.cos(a)
                y, x = y + dy, x + dx
                cell = (int(round(y)), int(round(x)))
                if cell in state["thrusters"]:
                    hits.append(state["thrusters"].index(cell)); continue
                if not state["inside"][cell] or (n > 10 and state["edge"][cell] < self.edge_band):
                    out.append(("crumb",) + cell)            # the far edge: now follow breadcrumbs
                else:
                    out.append(("ray", y, x, dy, dx, n + 1))
            else:
                _, y, x = pulse
                c, tag = state["crumb"][y, x], state["tag"]
                nxt = None
                for ddy in (-1, 0, 1):
                    for ddx in (-1, 0, 1):
                        yy, xx = y + ddy, x + ddx
                        if (ddy or ddx) and 0 <= yy < lab.SIZE and 0 <= xx < lab.SIZE and tag[yy, xx] == c:
                            nxt = (yy, xx)
                if nxt is None:
                    continue
                if nxt in state["thrusters"]:
                    hits.append(state["thrusters"].index(nxt))
                else:
                    out.append(("crumb",) + nxt)
        return out, hits

if __name__ == "__main__":
    for wobble in (0.0, 0.05, 0.1, 0.2):
        for seed in (0, 1):
            mech = Rays(wobble)
            st = mech.grow(seed)
            ok, wrong, lost, detail = lab.pulse_test(mech, st, *lab.layout(), repeats=5)
            print(f"3: rays, wobble {wobble}, seed {seed}: correct {ok:.2f}, crosstalk {wrong:.2f}, lost {lost:.2f}", flush=True)

class AimedRays(Rays):
    """Rays that steer up the edge-distance gradient (towards the centre) each step until they pass its peak, then fly
    straight: only the outward half is open-loop."""
    def step(self, state, pulses):
        e = state["edge"]
        fixed = []
        for pulse in pulses:
            if pulse[0] == "ray":
                _, y, x, dy, dx, n = pulse
                cy, cx = int(round(y)), int(round(x))
                gy = e[min(cy + 1, lab.SIZE - 1), cx] - e[max(cy - 1, 0), cx]
                gx = e[cy, min(cx + 1, lab.SIZE - 1)] - e[cy, max(cx - 1, 0)]
                if (gy * dy + gx * dx) > 0.3 and np.hypot(gy, gx) > 0.5:      # still climbing: re-aim at the peak
                    norm = np.hypot(gy, gx)
                    dy, dx = gy / norm, gx / norm
                pulse = ("ray", y, x, dy, dx, n)
            fixed.append(pulse)
        return Rays.step(self, state, fixed)
