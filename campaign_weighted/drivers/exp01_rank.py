"""Experiment 1 -- the rank is a speed parameter, and what the flow-weighted basis changes.

Sioux Falls and two Chicago Sketch pools; r in RANKS; both basis arms. Python (the engine in
which both arms are a single flag). Output: results/exp01_rank.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as K
import config as C


def main():
    rows = []
    for case in C.RANK_CASES:
        P, tau = K.load(case)
        reps = C.REPS_SMALL if P["n"] < 60000 else C.REPS_LARGE
        K.banner(f"exp01 {case}: n={P['n']:,} od={P['n_od']:,} tau={tau} reps={reps}")
        tfull, ofull, _ = K.solve_full_py(P, reps)
        print(f"  FULL {tfull:8.2f}s obj={ofull:,.2f}", flush=True)
        for rk in C.RANKS:
            for basis in C.BASES:
                Cc = K.compress(P, tau, rk, basis)
                t, o, _ = K.solve_comp_py(P, Cc, reps)
                gap = 100 * (o - min(o, ofull)) / min(o, ofull)
                print(f"  r={rk:3d} {basis:11s} {t:8.2f}s  speedup={tfull/t:5.2f}x  "
                      f"gap={gap:8.4f}%  svd={Cc['svd_time']:.2f}s", flush=True)
                rows.append(dict(case=case, n=P["n"], tau=tau, r=rk, basis=basis,
                                 reduction_pct=round(Cc["reduction_pct"], 1),
                                 t_full_s=round(tfull, 2), t_comp_s=round(t, 2),
                                 speedup=round(tfull / t, 2), gap_pct=round(gap, 4),
                                 svd_s=round(Cc["svd_time"], 2), obj=o, reps=reps))
                K.write("exp01_rank", rows)
    print(f"\nwrote {C.RESULTS/'exp01_rank.csv'}\n[exp01] DONE", flush=True)


if __name__ == "__main__":
    main()
