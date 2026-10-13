"""One-bit learning on a grown path, on a bare sheet. A red and a blue input cell side by side (sources), wired by the
breadcrumb rule to one thruster cell (sink). Pulses travel along the path, from cell to the cell its breadcrumb names.
Each input keeps a trace of its own recent pulses (fading, TRACE time). Visits, in random order: a burst of pulses at
one colour's input, then a taste at both inputs: bad after red (the poison colour), good after blue. A bad taste closes
the gate of an input whose trace is up (it stops being a source: the path is cut at its start). Measured: how many of
each colour's pulses reach the thruster, before and after learning."""
import sys, numpy as np, paths as P

TRACE = 15.0        # how long an input's trace of its own pulses lasts
PULSE_RATE = 0.3    # pulses per unit time at the input during a visit
VISIT = 20          # time units of pulses per visit
GAP = 70            # quiet time between the taste and the next visit

def run(seed=0, visits=12, poison="red", learn=True, verbose=False):
    rng = np.random.default_rng(seed + 100)
    red, blue, thruster = (22, 44), (26, 44), (24, 3)
    ends = P.Ends(sources=(red, blue), sinks=(thruster,))
    lives = P.life(seed, 1.0, ends)
    for t, p, now in lives:                                 # let the wiring grow first
        if t >= 300: break
    s = lives.gi_frame.f_locals["s"]; tag = lives.gi_frame.f_locals["tag"]
    n = P.SIZE
    fire = np.zeros((n, n), bool)
    trace = {"red": 0.0, "blue": 0.0}
    gate = {"red": True, "blue": True}
    arrived = {"red": [], "blue": []}                       # per visit: pulses that reached the thruster
    schedule = rng.permutation(["red", "blue"] * (visits // 2))
    for colour in schedule:
        cell = red if colour == "red" else blue
        got = 0
        for step in range(VISIT + GAP):
            t, p, now = next(lives)
            s = lives.gi_frame.f_locals["s"]
            path, crumb = s[P.PATH] > 0.5, s[P.CRUMB]
            # a pulse moves on, from each firing path cell to the neighbour its breadcrumb names
            nxt = np.zeros_like(fire)
            for y, x in np.argwhere(fire & path):
                c = crumb[y, x]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if (dy or dx) and 0 <= yy < n and 0 <= xx < n and tag[yy, xx] == c and path[yy, xx]:
                            nxt[yy, xx] = True
            if nxt[thruster]:
                got += 1
                nxt[thruster] = False
            fire = nxt
            for k in trace:
                trace[k] *= np.exp(-1.0 / TRACE)
            if step < VISIT and rng.random() < PULSE_RATE and gate[colour] and path[cell]:
                fire[cell] = True                           # a sensor pulse sets off down the path
                trace[colour] = 1.0
            if step == VISIT and learn:                      # the taste, at both inputs
                if colour == poison:
                    for k in gate:
                        if trace[k] > 0.5 and gate[k]:
                            gate[k] = False                 # bad taste while recently active: the gate closes
                            cuts = tuple(c for c, kk in ((red, "red"), (blue, "blue")) if not gate[kk])
                            lives.send(P.replace(now, cuts=cuts))
        arrived[colour].append(got)
        if verbose:
            print(f"visit {colour:4s}: {got} pulses reached the thruster; gates red {gate['red']} blue {gate['blue']}")
    return arrived, gate

if __name__ == "__main__":
    for seed in range(4):
        arrived, gate = run(seed, verbose=(seed == 0))
        r, b = arrived["red"], arrived["blue"]
        print(f"seed {seed}: red pulses arriving per visit {r}; blue {b}; gates {gate}", flush=True)
