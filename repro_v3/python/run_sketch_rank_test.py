"""Falsifiable prediction test on Chicago Sketch (author go, 2026-07-25).

PREDICTION (from the mechanism law of STORY 5b + the rank law of 5f): the whole Sketch axis
was run at r=50, i.e. 2r = 100 dense flops per minor row against only ~15 nnz per path --
exactly the wrong side of the crossover. Dropping to r = 10-20 (2r = 20-40) should RAISE the
Sketch speedup above the r=50 values (E0 1.95x, V2 0.98x in Python).

Cells: the two best Sketch pools, V2 (K-bar 2.45) and E0 (K-bar 3.49), tau = 4.54 fixed,
r in {10, 20, 50}. Both engines, fresh same-session full baselines so every ratio is
within-engine and within-session.

    python run_sketch_rank_test.py [--pools V2,E0] [--ranks 10,20,50]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CERTPY = ROOT / "source" / "updated_TAPLite" / "python"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CERTPY))

import v3_metrics as M
import compressed_assignment as ca
from run_kod_axis import POOLS, TAU
from run_grid_w0free import solve as cpp_solve

ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "sketch_rank"
TOL = 1e-4


def export(dsdir, pool, tau, rank, out_dir):
    if (out_dir / "meta.json").exists():
        return
    r = subprocess.run([sys.executable, str(CERTPY / "export_stage7.py"), str(dsdir), pool,
                        str(out_dir), str(rank), "1", str(tau)],
                       capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"export failed: {r.stderr[-300:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="V2,E0")
    ap.add_argument("--ranks", default="10,20,50")
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]
    out = HERE.parent / "results" / "sketch_rank_test.csv"
    rows = []

    def rec(**kw):
        rows.append(kw)
        keys = sorted({k for r_ in rows for k in r_})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r_ in rows:
                w.writerow({k: r_.get(k, "") for k in keys})

    for tag in [p.strip() for p in a.pools.split(",")]:
        dsdir, pool = POOLS[tag]
        print(f"\n=== {tag} (tau={TAU}) ===", flush=True)

        # ---------- Python ----------
        P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
        t = time.perf_counter(); sf = ca.solve_full(P, tol=TOL, max_outer=40)
        tpf = time.perf_counter() - t
        xf = M.convert_euclid(sf["x_raw"], P["d"], P["p2od"])
        opf = float(P["bpr"].beckmann(M.link_flow(P, xf)))
        print(f"  PY  full           t={tpf:7.1f}s obj={opf:,.2f}", flush=True)
        rec(pool=tag, engine="py", r="", t_s=round(tpf, 2), obj=opf, speedup="", gap_pct="")
        major = ca.split_major_minor(P, tau=TAU)
        for rk in ranks:
            C = ca.build_compressed(P, major, rk)
            t = time.perf_counter()
            sc = ca.solve_compressed(P, C, regime="hard", tol=TOL, max_outer=30)
            tph = time.perf_counter() - t
            xc = M.convert_euclid(sc["x_raw"], P["d"], P["p2od"])
            oph = float(P["bpr"].beckmann(M.link_flow(P, xc)))
            gap = 100 * (oph - min(oph, opf)) / min(oph, opf)
            print(f"  PY  hard r={rk:3d}      t={tph:7.1f}s SPEEDUP={tpf/tph:5.2f}x "
                  f"gap={gap:+.4f}%", flush=True)
            rec(pool=tag, engine="py", r=rk, t_s=round(tph, 2), obj=oph,
                speedup=round(tpf / tph, 2), gap_pct=round(gap, 4))

        # ---------- C++ ----------
        d0 = SCRATCH / tag / "tau0"
        export(dsdir, pool, 0.0, 0, d0)
        tcf, inf_f = cpp_solve(d0, "full", 0)
        ocf = float(inf_f["obj_feasible"])
        print(f"  C++ full           t={tcf:7.1f}s obj={ocf:,.2f}", flush=True)
        rec(pool=tag, engine="cpp", r="", t_s=round(tcf, 2), obj=ocf, speedup="", gap_pct="")
        for rk in ranks:
            dr = SCRATCH / tag / f"r{rk}"
            export(dsdir, pool, TAU, rk, dr)
            tch, inf_h = cpp_solve(dr, "hard", rk)
            och = float(inf_h["obj_feasible"])
            gap = 100 * (och - min(och, ocf)) / min(och, ocf)
            print(f"  C++ hard r={rk:3d}      t={tch:7.1f}s SPEEDUP={tcf/tch:5.2f}x "
                  f"gap={gap:+.4f}%", flush=True)
            rec(pool=tag, engine="cpp", r=rk, t_s=round(tch, 2), obj=och,
                speedup=round(tcf / tch, 2), gap_pct=round(gap, 4))

    print("\n[rank-test] SUMMARY (speedup by rank):")
    print(f"  {'pool':5s} {'engine':7s} " + "".join(f"r={r:<8}" for r in ranks))
    for tag in [p.strip() for p in a.pools.split(",")]:
        for eng in ("py", "cpp"):
            line = f"  {tag:5s} {eng:7s} "
            for rk in ranks:
                m_ = [r_ for r_ in rows if r_["pool"] == tag and r_["engine"] == eng
                      and r_["r"] == rk]
                line += f"{m_[0]['speedup']:<10.2f}" if m_ else " " * 10
            print(line)
    print(f"wrote {out}\n[rank-test] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
