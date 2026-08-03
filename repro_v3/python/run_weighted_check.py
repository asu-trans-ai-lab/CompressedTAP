"""Does the flow-weighted basis buy accuracy at fixed rank? (author switch, 2026-07-25)

The campaign so far used the plain SVD of B2. The flow-weighted variant takes the SVD of
diag(sqrt(f_minor)) B2, so the retained subspace minimises the FLOW-WEIGHTED reconstruction
loss and high-flow minor paths dominate the basis. This is the accuracy/rank lever.

Fast decisive test before re-running the campaign: Sioux Falls (tau=600) and Chicago Sketch
V2/E0 (tau=4.54), r in {10,20,50}, unweighted vs weighted, Python (the engine where both
are one flag). Reports feasible objective gap and speedup for each.

    python run_weighted_check.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "vendor"))

import v3_metrics as M
import compressed_assignment as ca
from run_kod_axis import POOLS

CASES = [
    ("sioux", ROOT / "data/submitted_v2/sioux", "pool.csv", 600.0),
    ("sketch-V2", POOLS["V2"][0], POOLS["V2"][1], 4.54),
    ("sketch-E0", POOLS["E0"][0], POOLS["E0"][1], 4.54),
]
RANKS = [10, 20, 50]
TOL = 1e-4


def main():
    out = HERE.parent / "results" / "weighted_check.csv"
    rows = []
    for name, dsdir, pool, tau in CASES:
        P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
        t = time.perf_counter(); sf = ca.solve_full(P, tol=TOL, max_outer=40)
        tfull = time.perf_counter() - t
        xf = M.convert_euclid(sf["x_raw"], P["d"], P["p2od"])
        ofull = float(P["bpr"].beckmann(M.link_flow(P, xf)))
        print(f"\n=== {name}: n={P['n']:,} tau={tau} | FULL {tfull:.2f}s "
              f"obj={ofull:,.2f} ===", flush=True)
        major = ca.split_major_minor(P, tau=tau)
        for rk in RANKS:
            line = {}
            for tag, w in (("unweighted", None), ("weighted", P["x0"])):
                C = ca.build_compressed(P, major, rk, weight_flows=w)
                t = time.perf_counter()
                sc = ca.solve_compressed(P, C, regime="hard", tol=TOL, max_outer=30)
                tc = time.perf_counter() - t
                xc = M.convert_euclid(sc["x_raw"], P["d"], P["p2od"])
                oc = float(P["bpr"].beckmann(M.link_flow(P, xc)))
                gap = 100 * (oc - min(oc, ofull)) / min(oc, ofull)
                line[tag] = (tc, gap, oc, C["svd_time"])
                rows.append(dict(case=name, r=rk, basis=tag, t_s=round(tc, 2),
                                 speedup=round(tfull / tc, 2), gap_pct=round(gap, 4),
                                 obj=oc, svd_s=round(C["svd_time"], 2),
                                 t_full_s=round(tfull, 2)))
            u, w_ = line["unweighted"], line["weighted"]
            print(f"  r={rk:3d}  unweighted {tfull/u[0]:5.2f}x gap={u[1]:7.4f}%   "
                  f"weighted {tfull/w_[0]:5.2f}x gap={w_[1]:7.4f}%   "
                  f"gap change {w_[1]-u[1]:+.4f} pp", flush=True)
            keys = list(rows[0].keys())
            with out.open("w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {out}\n[weighted-check] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
