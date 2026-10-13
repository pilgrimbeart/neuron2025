import numpy as np, seed, cell
from excite import medium
def channel_sheet(D=8, rise=0.5, gamma=2.0, seed_=0, settle=600):
    terms = medium(D=D)
    terms["w"]["v"] = rise
    terms["m"] = {"m": gamma, "nm": -gamma}
    Dv = terms["v"].pop("m*nv"); terms["v"].update({"m*nv": Dv / 2, "nv": Dv / 2})
    s = seed.Sheet(seed.rule(terms), seed=seed_, start={"v": 0.0, "w": 0.0, "t": 0.0})
    s.a[cell.VARIABLES.index("m"), :s.n] = np.random.default_rng(seed_).uniform(-0.1, 0.1, s.n)
    s.step(settle)
    return s, terms
def components(s, mask):
    """Connected pieces of the cells in mask (8-neighbour), as a label per cell (-1 outside)."""
    label = np.full(s.n, -1); k = 0
    nb = s.g.sheet.neighbours
    for i in range(s.n):
        if mask[i] and label[i] < 0:
            stack = [i]; label[i] = k
            while stack:
                j = stack.pop()
                for q in nb[j]:
                    if q < s.n and mask[q] and label[q] < 0:
                        label[q] = k; stack.append(q)
            k += 1
    return label
def wave(s, start, ticks=400):
    s.kick(start); fired = np.zeros(s.n, bool)
    for _ in range(ticks):
        s.step(); fired |= s.var("v") > 0.5
    return fired
if __name__ == "__main__":
    s, _ = channel_sheet(); m = s.var("m")
    lab = components(s, m > 0)
    sizes = np.bincount(lab[lab >= 0])
    src = s.g.inputs[0, 0]
    print(f"channel pieces: {len(sizes)}, sizes: largest {sizes.max()}, median {int(np.median(sizes))}; the input's piece: {sizes[lab[src]]} cells")
    big = np.argmax(sizes); start = np.nonzero(lab == big)[0][0]
    fired = wave(s, start)
    print(f"kick in the largest piece ({sizes[big]} cells): fired {fired.sum()}, of which in that piece {(fired & (lab == big)).sum()}")
