"""Mechanisms 1 and 2: breadcrumb paths (paths.py) in one or more channels; pulses follow breadcrumbs.
1: a channel per thruster, its sources the inputs of the opposite group (crossed). 2: one channel, every input a source,
every thruster a sink (each input wires to its nearest thruster: right only for pulling thrusters)."""
import sys, numpy as np
sys.path.insert(0, "/home/pilgrim/git/neuron2025")
import paths as P
import lab

class Channels:
    def __init__(self, crossed=True, grow_time=300):
        self.crossed, self.grow_time = crossed, grow_time
    def grow(self, seed=0):
        inputs, thrusters = lab.layout()
        if self.crossed:
            chans = [P.Ends(sources=tuple(inputs[i] for i in range(len(inputs)) if (i // 2 + lab.G // 2) % lab.G == k),
                            sinks=(t,)) for k, t in enumerate(thrusters)]
        else:
            chans = [P.Ends(sources=tuple(inputs), sinks=tuple(thrusters))]
        state = {"path": [], "crumb": [], "tag": [], "sources": [], "thrusters": thrusters}
        for i, ends in enumerate(chans):
            lives = P.life(seed + 17 * i, 1.0, ends)
            for t, p, now in lives:
                if t >= self.grow_time: break
            f = lives.gi_frame.f_locals
            state["path"].append(f["s"][P.PATH] > 0.5); state["crumb"].append(f["s"][P.CRUMB].copy()); state["tag"].append(f["tag"])
            state["sources"].append(set(ends.sources))
        return state
    def inject(self, state, cell):
        return [(k, cell) for k in range(len(state["path"])) if cell in state["sources"][k]]
    def step(self, state, pulses):
        out, hits = [], []
        for k, (y, x) in pulses:
            c, path, tag = state["crumb"][k][y, x], state["path"][k], state["tag"][k]
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (dy or dx) and 0 <= yy < lab.SIZE and 0 <= xx < lab.SIZE and tag[yy, xx] == c and path[yy, xx]:
                        nxt = (yy, xx)
            if nxt is None:
                continue                                    # the trail ends: lost
            if nxt in state["thrusters"]:
                hits.append(state["thrusters"].index(nxt))
            else:
                out.append((k, nxt))
        return out, hits

if __name__ == "__main__":
    for name, mech in (("1: channel per thruster (crossed)", Channels(True)), ("2: one channel (nearest thruster)", Channels(False))):
        for seed in (0, 1, 2):
            st = mech.grow(seed)
            ok, wrong, lost, detail = lab.pulse_test(mech, st, *lab.layout())
            print(f"{name}, seed {seed}: correct {ok:.2f}, crosstalk {wrong:.2f}, lost {lost:.2f}", flush=True)
