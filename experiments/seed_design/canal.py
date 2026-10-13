"""Canalisation test: pulse one input now and then; does m (conductance) grow along the routes activity takes?"""
import sys
import numpy as np, seed, cell
from excite import medium

def run(alpha, beta, m_rest, m_start, flank, ticks=6000, every=150, seed_=0, picture=None):
    terms = medium()
    terms["m"] = {"v*nv": alpha, "m": -beta, "1": beta * m_rest, "nv": -flank, "v*nv": alpha + flank}
    s = seed.Sheet(seed.rule(terms), seed=seed_, start={"v": 0.0, "w": 0.0, "t": 0.0})
    rng = np.random.default_rng(seed_)
    s.a[cell.VARIABLES.index("m"), :s.n] = m_start + rng.uniform(-0.05, 0.05, s.n)
    source = s.g.inputs[0, 0]
    shots = []
    for t in range(ticks):
        if t % every == 0:
            s.kick(source, 1.0)
            # a neighbour or two, so the pulse enters as a small front
        s.step()
        if picture and t in picture:
            shots.append((f"m t={t}", s.grid("m") * 2 - 1))
    m = s.var("m")
    return m, shots, s

if __name__ == "__main__":
    for alpha, flank in ((2.0, 0.0), (2.0, 1.0), (4.0, 2.0)):
        m, shots, s = run(alpha=alpha, beta=1.0, m_rest=0.5, m_start=0.85, flank=flank, picture={0, 1500, 3000, 5999})
        print(f"alpha {alpha} flank {flank}: m min {m.min():.2f} mean {m.mean():.2f} max {m.max():.2f}; cells above 0.7 (conducting): {(m > 0.7).sum()} of {s.n}")
        seed.picture(dict(shots), f"canal_a{alpha}_f{flank}.png", scale=4)
