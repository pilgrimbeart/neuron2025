"""Channels in m by anti-diffusion (dm = gamma (m - nm)): do waves then travel along them?"""
import numpy as np, seed, cell
from excite import medium

def make(gamma, seed_=0, settle=600):
    terms = medium()
    terms["m"] = {"m": gamma, "nm": -gamma}
    D = terms["v"].pop("m*nv")                           # conductance (1 + m) / 2: insulators (m = -1) are inert
    terms["v"].update({"m*nv": D / 2, "nv": D / 2})
    s = seed.Sheet(seed.rule(terms), seed=seed_, start={"v": 0.0, "w": 0.0, "t": 0.0})
    rng = np.random.default_rng(seed_)
    s.a[cell.VARIABLES.index("m"), :s.n] = rng.uniform(-0.1, 0.1, s.n)
    s.step(settle)
    return s

if __name__ == "__main__":
    for gamma in (2.0, 5.0):
        s = make(gamma)
        m = s.var("m")
        print(f"gamma {gamma}: after settling, m at +1: {(m > 0.9).sum()}, at -1: {(m < -0.9).sum()}, in between: {((m > -0.9) & (m < 0.9)).sum()}")
        shots = [("m", s.grid("m"))]
        s.kick(s.g.inputs[0, 0])
        fired = np.zeros(s.n, bool)
        for t in range(1, 201):
            s.step(); fired |= s.var("v") > 0.5
            if t in (20, 60, 120, 200):
                shots.append((f"v t={t}", s.grid("v")))
        print(f"   one kick at the east red input: cells fired {fired.sum()}; of those on m>0 channels {(fired & (m > 0)).sum()}")
        shots.append(("fired", s.g.grid(fired.astype(float) * 2 - 1)))
        seed.picture(dict(shots), f"channels_g{gamma}.png", scale=4)
