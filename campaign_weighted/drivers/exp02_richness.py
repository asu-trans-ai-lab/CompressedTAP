"""Experiment 2 -- route-richness axis on Chicago Sketch (same network and demand).

Pools of increasing average richness Kbar at fixed tau, at the rank chosen by the cost rule.
Both basis arms. Python. Output: results/exp02_richness.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as K
import config as C

RANK = 20


def main():
    rows = []
    for case in C.RICHNESS_AXIS:
        P, tau = K.load(case)
        kbar = P["n"] / P["n_od"]
        reps = C.REPS_SMALL if P["n"] < 60000 else C.REPS_LARGE
        K.banner(f"exp02 {case}: n={P['n']:,} Kbar={kbar:.2f} reps={reps}")
        tfull, ofull, _ = K.solve_full_py(P, reps)
        print(f"  FULL {tfull:8.1f}s obj={ofull:,.2f}", flush=True)
        for basis in C.BASES:
            Cc = K.compress(P, tau, RANK, basis)
            t, o, _ = K.solve_comp_py(P, Cc, reps)
            gap = 100 * (o - min(o, ofull)) / min(o, ofull)
            print(f"  {basis:11s} {t:8.1f}s speedup={tfull/t:5.2f}x gap={gap:8.4f}%",
                  flush=True)
            rows.append(dict(case=case, n=P["n"], ell=P["n_od"], kbar=round(kbar, 2),
                             tau=tau, r=RANK, basis=basis,
                             reduction_pct=round(Cc["reduction_pct"], 1),
                             t_full_s=round(tfull, 1), t_comp_s=round(t, 1),
                             speedup=round(tfull / t, 2), gap_pct=round(gap, 4),
                             svd_s=round(Cc["svd_time"], 1), obj=o, reps=reps))
            K.write("exp02_richness", rows)
    print(f"\nwrote {C.RESULTS/'exp02_richness.csv'}\n[exp02] DONE", flush=True)


if __name__ == "__main__":
    main()
