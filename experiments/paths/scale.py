"""paths.py at a bigger size: one pair across the sheet, and 3 sources with 2 sinks, scaled. When each joins, whether
the sheet then falls quiet, and whether removing every sink clears the paths."""
import sys, time, numpy as np
import paths as P
def run(L, ends, remove_at, end, seed=0):
    P.SIZE = L
    lives = P.life(seed, 1.0, ends)
    joined_at, quiet, t0, log = None, [], time.time(), []
    for t, p, now in lives:
        s = lives.gi_frame.f_locals["s"]
        if t % 50 == 0 and t < remove_at:
            ok = all(any(P.route(p, a, k) is not None for k in now.sinks) for a in now.sources)
            if ok and joined_at is None: joined_at = t
            if joined_at is not None and t > joined_at + 0.6 * (remove_at - joined_at): quiet.append((s[P.EXCITED_LEFT] > 0).mean())
        if t == remove_at:
            log.append(f"path cells {(p > 0.5).sum()}")
            t, p, now = lives.send(P.replace(now, sinks=()))
        if t >= end:
            log.append(f"after removing the sinks: {(p > 0.5).sum()} path cells left at {end:.0f}")
            break
    return f"joined at {joined_at}; wave activity once wired {np.mean(quiet) if quiet else float('nan'):.3f}; " + "; ".join(log) + f"  ({(time.time() - t0) / end * 1000:.1f} ms/tick)"
if __name__ == "__main__":
    L = int(sys.argv[1])
    m, q = L // 2, L // 6
    print(f"{L}x{L}, one pair across:", run(L, P.Ends(sources=((m, L - 3),), sinks=((m, 2),)), 8 * L + 2000, 8 * L + 3600), flush=True)
    print(f"{L}x{L}, 3 sources 2 sinks:", run(L, P.Ends(sources=((q, L - 3), (m, L - 3), (L - q, L - 3)), sinks=((L // 4, 2), (3 * L // 4, 2))), 9 * L + 2500, 9 * L + 4100), flush=True)
