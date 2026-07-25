"""Step 3 of the central case: can the falling branch (K-bar >= 8.6, speedup ~0.66x) be
fixed WITHIN the paper's framework?

Two candidate fixes, tested at the two falling-branch pools (E2, K10):
  (a) SMALL RANK: signed-SVD hard with r in {10, 25} -- the dense penalty is O(n_minor * r),
      so cutting r cuts the dominant cost linearly. Table 5 says quality is insensitive to r.
  (b) GROUPED BOX-BOUND (major_minor_rg.MMModel + solve_reduced): minors in fixed nonneg
      groups, feasibility is a plain box bound s>=0 -- NO dense reconstruction at all,
      every matvec sparse. The structural fix.

Full-ALM baselines are the ones timed in the SAME session's axis sweep (kod_axis_alm.csv);
speedups are reported against them. Feasible objectives via the shared v3_metrics conversion.

    python run_highk_fixes.py [--pools E2,K10] [--ranks 10,25] [--m0 3] [--r0 1]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "source/updated_TAPLite/python"))

import v3_metrics as M
import compressed_assignment as ca
from major_minor_rg import MMModel, solve_reduced
from run_kod_axis import POOLS, TAU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="E2,K10")
    ap.add_argument("--ranks", default="10,25")
    ap.add_argument("--m0", type=int, default=3)
    ap.add_argument("--r0", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    ranks = [int(r) for r in a.ranks.split(",")]

    base = {r["pool"]: r for r in csv.DictReader(
        (HERE.parent / "results" / "kod_axis_alm.csv").open())}
    out = HERE.parent / "results" / "highk_fixes.csv"
    rows = []

    def flush():
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    for tag in [p.strip() for p in a.pools.split(",")]:
        dsdir, pool = POOLS[tag]
        b = base[tag]
        t_full = float(b["t_full_s"]); f_full = float(b["obj_full"])
        print(f"\n[fix] {tag}: baseline full t={t_full:.1f}s obj={f_full:,.2f}", flush=True)
        P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
        sl = None

        def feas_obj(x_raw):
            xf = M.convert_euclid(np.asarray(x_raw, float), P["d"], P["p2od"])
            return float(P["bpr"].beckmann(M.link_flow(P, xf)))

        # (a) small-rank signed-SVD hard
        major = ca.split_major_minor(P, tau=TAU)
        for r in ranks:
            C = ca.build_compressed(P, major, r)
            t = time.perf_counter()
            sc = ca.solve_compressed(P, C, regime="hard", tol=a.tol, max_outer=30)
            secs = time.perf_counter() - t
            obj = feas_obj(sc["x_raw"])
            gap = 100 * (obj - min(obj, f_full)) / min(obj, f_full)
            su = t_full / secs
            print(f"  hard r={r:3d}: t={secs:7.1f}s SPEEDUP={su:5.2f}x gap={gap:+.4f}% "
                  f"(svd {C['svd_time']:.1f}s)", flush=True)
            rows.append(dict(pool=tag, fix=f"svd_r{r}", t_s=round(secs, 1),
                             speedup=round(su, 2), obj_feasible=obj,
                             gap_vs_best_pct=round(gap, 4), t_full_s=t_full,
                             extra=f"svd_time={C['svd_time']:.1f}"))
            flush()

        # (b) grouped box-bound reduced gradient
        t = time.perf_counter()
        model = MMModel(P, m0=a.m0, r0=a.r0)
        build_s = time.perf_counter() - t
        t = time.perf_counter()
        rg = solve_reduced(P, model, tol=a.tol)
        secs = time.perf_counter() - t
        obj = feas_obj(rg["x"])
        gap = 100 * (obj - min(obj, f_full)) / min(obj, f_full)
        su = t_full / secs
        print(f"  grouped m0={a.m0} r0={a.r0}: t={secs:7.1f}s SPEEDUP={su:5.2f}x "
              f"gap={gap:+.4f}% n_red={rg['n_red']:,} (build {build_s:.1f}s)", flush=True)
        rows.append(dict(pool=tag, fix=f"grouped_m{a.m0}r{a.r0}", t_s=round(secs, 1),
                         speedup=round(su, 2), obj_feasible=obj,
                         gap_vs_best_pct=round(gap, 4), t_full_s=t_full,
                         extra=f"n_red={rg['n_red']},build={build_s:.1f}s"))
        flush()

    print("\n[fix] SUMMARY:")
    for r in rows:
        print(f"  {r['pool']:4s} {r['fix']:14s} SPEEDUP={r['speedup']:5.2f}x "
              f"gap={r['gap_vs_best_pct']:+.4f}%")
    print(f"wrote {out}\n[fix] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
