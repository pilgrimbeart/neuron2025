import sys, numpy as np
import expH
from sim import *
from expE import run
from expG import error
from multiprocessing import Pool
expH.NEAR = 0.45
expH.MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
def job(a):
    n, s, sigma = a
    P1 = run(n, s, sigma, 1.7); P2 = run(n, s + 1000, sigma, 1.7, cross=P1, c=1.0, form='saturate')
    z = expH.coords(P1, P2, np.random.default_rng(s + 5000), sigma, 40 * n * n)
    return error(z), unfolded(z)[0]
if __name__ == "__main__":
    for n in (int(a) for a in sys.argv[1].split(",")):
        for sigma in (1e-3, 1e-2):
            with Pool(8) as pool:
                r = pool.map(job, [(n, s, sigma) for s in range(16)])
            e = np.array([x[0] for x in r])
            print(f"I n {n} noise {sigma}: right (error < 0.1) {(e < 0.1).sum()}/16, unfolded {sum(x[1] for x in r)}/16; errors {np.round(np.sort(e), 3)}", flush=True)
