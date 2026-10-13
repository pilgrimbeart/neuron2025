import numpy as np, seed, cell
def medium(A=4.0, THETA=0.3, B=10.0, D=4.0, decay=0.2):
    return {"v": {"v^2": A, "v": -A*THETA, "v*w": -B, "m*nv": D}, "w": {"v": 1.0, "w": -decay}}
def test(terms, m0, ticks=500, dt=1.0):
    s = seed.Sheet(seed.rule(terms), start={"v": 0.0, "w": 0.0, "m": m0, "t": 0.0}, dt=dt)
    c = s.g.centre[0]; s.kick(c, 1.0)
    reached = np.zeros(s.n, bool); first = np.full(s.n, -1); spikes = np.zeros(s.n, int); was = np.zeros(s.n, bool)
    for t in range(int(ticks / dt)):
        s.step(); v = s.var("v"); up = v > 0.5
        spikes += up & ~was; was = up
        newly = up & ~reached; first[newly] = t; reached |= up
    far = np.hypot(*(s.g.xy[reached] - s.g.xy[c]).T).max() if reached.any() else 0
    return reached.sum(), far, first.max() * dt, (np.abs(s.var('v')) > 0.05).sum(), spikes.max()
if __name__ == "__main__":
    for decay in (0.2, 0.1):
        for m0 in (0.3, 0.4, 0.5, 0.6, 0.8):
            a = test(medium(decay=decay), m0)
            print(f"w decay {decay}, m {m0}: reached {a[0]:4d} (farthest {a[1]:.0f}, by time {a[2]:.0f}), active at end {a[3]:4d}, most spikes by one cell {a[4]}")
