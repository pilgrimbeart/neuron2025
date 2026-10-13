"""Mechanism 4: a hub (chiasm). One channel, one sink at the centre; every input and every thruster grows a path to it
(sources), so all paths converge on the hub. A pulse travels in along breadcrumbs, carrying its heading (the direction
of its last step); at the hub and beyond it travels out against breadcrumbs (to a path cell whose breadcrumb names the
current cell), choosing at each branch the one that best keeps its heading. Arriving at a thruster counts."""
import sys, numpy as np
sys.path.insert(0, "/home/pilgrim/git/neuron2025")
import paths as P
import lab

class Hub:
    def grow(self, seed=0):
        inputs, thrusters = lab.layout()
        hub = (int(lab.C), int(lab.C))
        lives = P.life(seed, 1.0, P.Ends(sources=tuple(inputs) + tuple(thrusters), sinks=(hub,)))
        for t, p, now in lives:
            if t >= 300: break
        f = lives.gi_frame.f_locals
        return {"path": f["s"][P.PATH] > 0.5, "crumb": f["s"][P.CRUMB].copy(), "tag": f["tag"], "hub": hub,
                "thrusters": thrusters}
    def inject(self, state, cell):
        return [("in", cell, (0.0, 0.0))]
    def step(self, state, pulses):
        out, hits = [], []
        path, crumb, tag = state["path"], state["crumb"], state["tag"]
        for kind, (y, x), (hy, hx) in pulses:
            nbrs = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)
                    and 0 <= y + dy < lab.SIZE and 0 <= x + dx < lab.SIZE and path[y + dy, x + dx]]
            if kind == "in" and (y, x) != state["hub"]:
                nxt = [c for c in nbrs if tag[c] == crumb[y, x]]
                if not nxt:
                    continue
                ny, nx_ = nxt[0]
                heading = (ny - y, nx_ - x)
                kind2 = "out" if nxt[0] == state["hub"] else "in"
                out.append((kind2, nxt[0], heading)); continue
            # outwards: to path cells whose breadcrumb names this cell, keeping the heading
            children = [c for c in nbrs if crumb[c] == tag[y, x]]
            if not children:
                continue
            norm = np.hypot(hy, hx) or 1.0
            best = max(children, key=lambda c: ((c[0] - y) * hy + (c[1] - x) * hx) / (norm * np.hypot(c[0] - y, c[1] - x)))
            if best in state["thrusters"]:
                hits.append(state["thrusters"].index(best)); continue
            new_heading = (0.7 * hy + 0.3 * (best[0] - y), 0.7 * hx + 0.3 * (best[1] - x))
            out.append(("out", best, new_heading))
        return out, hits

if __name__ == "__main__":
    for seed in (0, 1, 2):
        mech = Hub(); st = mech.grow(seed)
        ok, wrong, lost, detail = lab.pulse_test(mech, st, *lab.layout(), repeats=1)
        print(f"4: hub (chiasm), seed {seed}: correct {ok:.2f}, crosstalk {wrong:.2f}, lost {lost:.2f}", flush=True)
        if seed == 0:
            for i, target, got in detail: print("   input", i, "target", target, "reached", got)

class Hub2(Hub):
    """Two channels: inputs grow paths to the hub in one, thrusters in the other. A pulse comes in on the first and
    leaves on the second (against its breadcrumbs), keeping its heading."""
    def grow(self, seed=0):
        inputs, thrusters = lab.layout()
        hub = (int(lab.C), int(lab.C))
        st = {"hub": hub, "thrusters": thrusters}
        for name, src, sd in (("in", inputs, seed), ("out", thrusters, seed + 17)):
            lives = P.life(sd, 1.0, P.Ends(sources=tuple(src), sinks=(hub,)))
            for t, p, now in lives:
                if t >= 300: break
            f = lives.gi_frame.f_locals
            st[name] = (f["s"][P.PATH] > 0.5, f["s"][P.CRUMB].copy(), f["tag"])
        return st
    def step(self, state, pulses):
        out, hits = [], []
        for kind, (y, x), (hy, hx) in pulses:
            path, crumb, tag = state["in" if kind == "in" else "out"]
            nbrs = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)
                    and 0 <= y + dy < lab.SIZE and 0 <= x + dx < lab.SIZE and path[y + dy, x + dx]]
            if kind == "in":
                nxt = [c for c in nbrs if tag[c] == crumb[y, x]]
                if not nxt:
                    continue
                heading = (0.5 * hy + (nxt[0][0] - y), 0.5 * hx + (nxt[0][1] - x))
                out.append(("out" if nxt[0] == state["hub"] else "in", nxt[0], heading)); continue
            ptag = state["out"][2]
            children = [c for c in nbrs if crumb[c] == ptag[y, x]]
            if not children:
                continue
            norm = np.hypot(hy, hx) or 1.0
            best = max(children, key=lambda c: ((c[0] - y) * hy + (c[1] - x) * hx) / (norm * np.hypot(c[0] - y, c[1] - x)))
            if best in state["thrusters"]:
                hits.append(state["thrusters"].index(best)); continue
            out.append(("out", best, (0.7 * hy + 0.3 * (best[0] - y), 0.7 * hx + 0.3 * (best[1] - x))))
        return out, hits
