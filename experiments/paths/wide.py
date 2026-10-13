import sys, numpy as np, suite, json
from multiprocessing import Pool
def one(args):
    seed, dt, overrides = args
    import paths as P, suite as S
    for k, v in overrides.items(): setattr(P, k, v)
    out = []
    for name, (ends, change) in S.SCENES.items():
        lives = P.life(seed, dt, ends); res = {}
        for t, p, now in lives:
            if abs(t - 300) < dt / 2:
                res["before"] = S.joined_fraction(p, now); t, p, now = lives.send(change(now))
            if abs(t - 600) < dt / 2:
                res["mid"] = S.joined_fraction(p, now)
            if t >= 900 - dt / 2:
                res["end"] = S.joined_fraction(p, now); res["cells"] = int((p > 0.5).sum()); break
        out.append((name, res))
    return out
if __name__ == "__main__":
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    with Pool(8) as pool:
        results = pool.map(one, [(s, dt, overrides) for s in range(4) for dt in (1.0, 0.5)])
    by = {}
    for r in results:
        for name, res in r: by.setdefault(name, []).append(res)
    m = lambda xs: f"{np.mean(xs):.2f}" if xs else "  - "
    for name, rs in by.items():
        g = lambda k: [x[k] for x in rs if x.get(k) is not None]
        print(f"{name:40s} joined at 300: {m(g('before'))}; 300 after the change: {m(g('mid'))}; 600 after: {m(g('end'))}; cells at the end {np.mean(g('cells')):.0f}")
