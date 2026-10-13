"""Crossing paths, approach 2: one shared excitable medium; each wave carries its sink's label (0..L-1), and each cell
keeps a breadcrumb per label. Sources each target one label. Otherwise the breadcrumb rule as in paths.py."""
import sys, numpy as np
from numba import njit
sys.path.insert(0, ".")
import paths as P

@njit(cache=False)
def ticks(ex, rest, count, label, crumb, via, best, bcrumb, age, path, tag, sink_label, src_target, rng_seed, steps,
          excited, refractory, expire, update, jitter, margin, rate, L):
    np.random.seed(rng_seed)
    n = ex.shape[0]
    root2 = np.sqrt(2.0)
    far = 1000.0
    for _ in range(steps):
        for y in range(n):
            for x in range(n):
                for l in range(L):
                    age[l, y, x] += 1.0
                if ex[y, x] > 0.0:
                    ex[y, x] -= 1.0
                    if ex[y, x] <= 0.0:
                        ex[y, x] = 0.0
                        rest[y, x] = refractory
                        l = int(label[y, x])
                        if l >= 0 and sink_label[y, x] < 0 and via[l, y, x] > best[l, y, x] * (1 + margin) and bcrumb[l, y, x] >= 0:
                            crumb[l, y, x] = bcrumb[l, y, x]
                elif rest[y, x] > 0.0:
                    rest[y, x] = max(0.0, rest[y, x] - 1.0)
                if sink_label[y, x] >= 0 and np.random.random() < rate and ex[y, x] == 0.0 and rest[y, x] == 0.0:
                    l = int(sink_label[y, x])
                    ex[y, x], count[y, x], label[y, x] = excited, 0.0, l
                    age[l, y, x] = 0.0
                    crumb[l, y, x] = -1.0
        ex2, count2, label2 = ex.copy(), count.copy(), label.copy()
        rest2 = rest.copy()
        crumb2, via2, best2, bcrumb2, age2, path2 = crumb.copy(), via.copy(), best.copy(), bcrumb.copy(), age.copy(), path.copy()
        for y in range(n):
            for x in range(n):
                if np.random.random() >= update:
                    continue
                wave, wcrumb, wlabel = far, -1.0, -1
                mine = int(label[y, x]) if ex[y, x] > 0.0 else -1
                via_c = far
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = y + dy, x + dx
                        if not ((dy or dx) and 0 <= yy < n and 0 <= xx < n):
                            continue
                        if ex[yy, xx] > 0.0:
                            lj = int(label[yy, xx])
                            if mine >= 0 and lj != mine:
                                continue                                    # another sink's wave: not mine to watch
                            v = count[yy, xx] + (root2 if dy and dx else 1.0)
                            if v < wave:
                                wave, wcrumb, wlabel = v, tag[yy, xx], lj
                            if mine >= 0 and tag[yy, xx] == crumb[mine, y, x] and count[yy, xx] < count[y, x]:
                                via_c = v
                for l in range(L):                                          # path, per label
                    named = False
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            yy, xx = y + dy, x + dx
                            if (dy or dx) and 0 <= yy < n and 0 <= xx < n and path[l, yy, xx] > 0.5 and crumb[l, yy, xx] == tag[y, x]:
                                named = True
                    path2[l, y, x] = 1.0 if (src_target[y, x] == l or named) else 0.0
                    if age[l, y, x] >= expire:
                        crumb2[l, y, x] = -1.0
                if ex[y, x] > 0.0:
                    if wave < count[y, x] and count[y, x] > 0.0:
                        count2[y, x] = wave
                    if via_c < via[mine, y, x]:
                        via2[mine, y, x] = via_c
                    if wave < best[mine, y, x]:
                        best2[mine, y, x], bcrumb2[mine, y, x] = wave, wcrumb
                elif rest[y, x] == 0.0 and wave < far:
                    ex2[y, x], count2[y, x], label2[y, x] = excited, wave + jitter * np.random.standard_normal(), wlabel
                    age2[wlabel, y, x] = 0.0
                    # the breadcrumb watch for this label starts: via my crumb (if it is among the exciters), best so far
                    vc = far
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            yy, xx = y + dy, x + dx
                            if (dy or dx) and 0 <= yy < n and 0 <= xx < n and ex[yy, xx] > 0.0 and int(label[yy, xx]) == wlabel \
                                    and tag[yy, xx] == crumb[wlabel, y, x]:
                                vc = count[yy, xx] + (root2 if dy and dx else 1.0)
                    via2[wlabel, y, x], best2[wlabel, y, x], bcrumb2[wlabel, y, x] = vc, wave, wcrumb
        ex[:], count[:], label[:], rest[:] = ex2, count2, label2, rest2
        crumb[:], via[:], best[:], bcrumb[:], age[:], path[:] = crumb2, via2, best2, bcrumb2, age2, path2

def run(pairs, seed=0, steps=400, report_every=100):
    n, L = P.SIZE, len(pairs)
    rng = np.random.default_rng(seed)
    tag = rng.random((n, n))
    ex, rest, count, label = np.zeros((n, n)), np.zeros((n, n)), np.full((n, n), 1000.0), np.full((n, n), -1.0)
    crumb, via, best, bcrumb = [np.full((L, n, n), v) for v in (-1.0, 1000.0, 1000.0, -1.0)]
    age, path = np.full((L, n, n), 1000.0), np.zeros((L, n, n))
    sink_label, src_target = np.full((n, n), -1.0), np.full((n, n), -1.0)
    for l, (src, snk) in enumerate(pairs):
        sink_label[snk], src_target[src] = l, l
    out = []
    for k in range(steps // report_every):
        ticks(ex, rest, count, label, crumb, via, best, bcrumb, age, path, tag, sink_label, src_target, seed * 1000 + k,
              report_every, P.EXCITED, P.REFRACTORY, P.EXPIRE, P.UPDATE, P.JITTER, P.MARGIN, P.RATE, L)
        joined = [P.route(path[l], src, snk) is not None for l, (src, snk) in enumerate(pairs)]
        out.append(f"t{(k + 1) * report_every}: joined {sum(joined)}/{L}, cells {int((path > 0.5).any(axis=0).sum())}")
    return out, path

if __name__ == "__main__":
    S = P.SIZE
    for name, pairs in (("two crossing", [((6, 3), (S - 7, S - 4)), ((S - 7, 3), (6, S - 4))]),
                        ("six on a ring to the opposite side", [((int(24 + 21 * np.sin(a)), int(24 + 21 * np.cos(a))),
                                                                  (int(24 - 21 * np.sin(a)), int(24 - 21 * np.cos(a))))
                                                                 for a in np.arange(6) * np.pi / 3])):
        for seed in (0, 1):
            out, path = run(pairs, seed)
            print(f"{name}, seed {seed}: " + " | ".join(out), flush=True)
