"""Fewer, smaller channel pieces (a bias towards insulating): does the bad-taste cut become colour-specific?"""
import numpy as np, seed, cell
from excite import medium
from chan2 import components
from learn import reach

def sheet(bias, eta=300.0, spread=0.5, fade=0.6, seed_=0, settle=600):
    terms = medium(D=8)
    terms["w"]["v"] = 0.5
    Dv = terms["v"].pop("m*nv"); terms["v"].update({"m*nv": Dv / 2, "nv": Dv / 2})
    terms["m"] = {"m": 2.0, "nm": -2.0, "1": -bias, "w*t": eta}
    terms["t"] = {"nt": spread, "t": -fade}
    s = seed.Sheet(seed.rule(terms), seed=seed_, start={"v": 0.0, "w": 0.0, "t": 0.0})
    s.a[cell.VARIABLES.index("m"), :s.n] = np.random.default_rng(seed_).uniform(-0.1, 0.1, s.n)
    s.step(settle)
    return s

if __name__ == "__main__":
    for bias in (0.0, 0.3, 0.6, 1.0):
        rows = []
        for seed_ in range(4):
            s = sheet(bias, seed_=seed_); g = s.g; m = s.var("m")
            lab = components(s, m > 0); sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.array([0])
            specific = 0; tried = 0
            for grp in range(6):                     # each sensor group: poison after red; does red's reach fall, blue's not?
                red, blue, taste = g.inputs[grp, 0], g.inputs[grp, 1], g.taste[grp]
                r0, b0 = reach(s, red), reach(s, blue)
                if r0 < 3 or b0 < 3:
                    continue
                tried += 1
                a, b = s.a.copy(), s.b.copy()
                s.kick(red); s.step(20); s.kick(taste, -1.0, var="t"); s.step(300)
                r1, b1 = reach(s, red), reach(s, blue)
                specific += (r1 < 0.5 * r0) and (b1 > 0.8 * b0)
                s.a[:], s.b[:] = a, b
            rows.append((float((m > 0).mean()), int(sizes.max()), float(np.median(sizes)), tried, specific))
        f, big, med, tried, spec = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]), sum(r[3] for r in rows), sum(r[4] for r in rows)
        print(f"bias {bias}: channel fraction {f:.2f}, largest piece {big:.0f}, median piece {med:.0f}; groups where both inputs connect: {tried}, cut red but spared blue: {spec}")
