"""THE CENTRAL CASE (author lock, 2026-07-24): Chicago Sketch, K/OD increasing.

One network, one demand, one axis: candidate-pool richness K-bar = n/ell. Everything else
fixed: tau=4.54, r=50, tol=1e-4, single thread, Python solvers (the certified engine for
timing). Phase 1 = ALM full vs ALM compressed-hard at every K/OD point -> the headline
speedup-vs-K/OD curve. (Operator rows GP/RG/FW attach to the same instances afterwards.)

Pools (all on disk, same network + demand.csv, only route richness differs):
    V2   sketch submitted pool        K-bar ~ 2.45
    E0   baseline enrichment          K-bar ~ 3.49
    E2   full rich pool               K-bar ~ 8.64
    K10  penalty-KSP depth 10         richer
    K15  penalty-KSP depth 15         richer still

    python run_kod_axis.py [--pools V2,E0,E2,K10,K15] [--tol 1e-4]
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
CERTPY = ROOT / "source" / "updated_TAPLite" / "python"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CERTPY))

import v3_metrics as M
import compressed_assignment as ca

SK_DATA = CERTPY.parent / "data" / "03_chicago_sketch"
V2_DIR = ROOT / "OR_paper_revision_V2" / "m4_rerun" / "data" / "sketch"
POOLS = {
    "V2":  (V2_DIR, "pool.csv"),
    "E0":  (SK_DATA, "path_pool_E0_baseline.csv"),
    "E2":  (SK_DATA, "path_pool.csv"),          # byte-identical to E2_full_rich
    "K10": (SK_DATA, "path_pool_K10.csv"),
    "K15": (SK_DATA, "path_pool_K15.csv"),
    "K20": (SK_DATA, "path_pool_K20.csv"),
}
TAU, RANK = 4.54, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="V2,E0,E2,K10,K15")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--max-outer-full", type=int, default=40)
    ap.add_argument("--max-outer-comp", type=int, default=30)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rank_use = a.rank

    out = (HERE.parent / "results" / (a.out or "kod_axis_alm.csv"))
    rows = []

    def flush():
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    for tag in [p.strip() for p in a.pools.split(",") if p.strip()]:
        dsdir, pool = POOLS[tag]
        print(f"\n[K-axis] {tag}: loading {pool}", flush=True)
        P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
        ell, n = P["n_od"], P["n"]
        kbar = n / ell
        major = ca.split_major_minor(P, tau=TAU)
        C = ca.build_compressed(P, major, rank_use)
        s = int(major.sum())
        red = 100 * (n - s - C["r"]) / n
        print(f"  n={n:,} ell={ell:,} K-bar={kbar:.2f} majors={s:,} r={C['r']} "
              f"red={red:.1f}% svd={C['svd_time']:.1f}s", flush=True)

        t = time.perf_counter()
        sf = ca.solve_full(P, tol=a.tol, max_outer=a.max_outer_full)
        t_full = time.perf_counter() - t
        xf = M.convert_euclid(sf["x_raw"], P["d"], P["p2od"])
        f_full = float(P["bpr"].beckmann(M.link_flow(P, xf)))
        print(f"  FULL t={t_full:8.1f}s obj={f_full:,.2f}", flush=True)

        t = time.perf_counter()
        sc = ca.solve_compressed(P, C, regime="hard", tol=a.tol,
                                 max_outer=a.max_outer_comp)
        t_hard = time.perf_counter() - t
        xc = M.convert_euclid(sc["x_raw"], P["d"], P["p2od"])
        f_comp = float(P["bpr"].beckmann(M.link_flow(P, xc)))
        f_ref = min(f_full, f_comp)
        gap = 100 * (f_comp - f_ref) / f_ref
        su = t_full / t_hard
        print(f"  HARD t={t_hard:8.1f}s SPEEDUP={su:5.2f}x objgap={gap:+.4f}% "
              f"(full-vs-best {100*(f_full-f_ref)/f_ref:+.4f}%)", flush=True)

        rows.append(dict(pool=tag, n=n, ell=ell, kbar=round(kbar, 2), majors=s,
                         rank=C["r"], reduction_pct=round(red, 1),
                         t_full_s=round(t_full, 1), t_hard_s=round(t_hard, 1),
                         speedup=round(su, 2), obj_full=f_full, obj_comp=f_comp,
                         gap_comp_vs_best_pct=round(gap, 4),
                         svd_time_s=round(C["svd_time"], 1)))
        flush()

    print("\n[K-axis] CENTRAL CURVE (speedup vs K/OD):")
    for r in rows:
        print(f"  {r['pool']:4s} K-bar={r['kbar']:6.2f} red={r['reduction_pct']:5.1f}% "
              f"speedup={r['speedup']:5.2f}x gap={r['gap_comp_vs_best_pct']:+.4f}%")
    print(f"wrote {out}\n[K-axis] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
