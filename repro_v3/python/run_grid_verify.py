"""Cross-engine / cross-rank verification suite on the grid (author question 2026-07-25:
"are those results verifiable using both Python and C++, and do they agree across K/OD
and ranks?").

Two home-regime cells (N16:K32, N32:K64). Per cell:
  (a) C++ rank sweep  : hard at r in {10,20,40} on fresh exports  -> rank sensitivity, C++
  (b) Python ALM      : solve_full + solve_compressed hard (r=20), same tol as C++ campaign
  (c) Python w0-free  : same compressed solve with C['x0m']=0 and v_base/d_eff moved
                        consistently -> does the C++ w0 finding replicate in Python?

Everything from the same pools/exports as grid_axis_cpp (regenerated fresh).
Writes results/grid_verify.csv.
"""
from __future__ import annotations

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
from run_grid_axis_cpp import gen_grid, logit_x0, KSP, ENV, SCRATCH as GSCRATCH
from run_grid_w0free import make_w0free, solve as cpp_solve

SOLVER_RANKS = [10, 20, 40]
CELLS = [(16, 32), (32, 64)]
TOL = 1e-4


def export(gdir, pool_name, tau, rank, out_dir):
    if (out_dir / "meta.json").exists():
        return
    r = subprocess.run([sys.executable, str(CERTPY / "export_stage7.py"), str(gdir),
                        pool_name, str(out_dir), str(rank), "1", str(tau)],
                       capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"export failed: {r.stderr[-300:]}")


def main():
    out = HERE.parent / "results" / "grid_verify.csv"
    rows = []

    def rec(**kw):
        rows.append(kw)
        keys = sorted({k for r_ in rows for k in r_})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r_ in rows:
                w.writerow({k: r_.get(k, "") for k in keys})

    for N, K in CELLS:
        cell = f"N{N}K{K}"
        gdir = GSCRATCH / f"g{N}"
        m, n_od, q = gen_grid(N, gdir)
        pool = gdir / f"verify_pool_K{K}.csv"
        r = subprocess.run([str(KSP), str(gdir), str(K), "1.4", str(pool)],
                           capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, r.stderr[-200:]
        df = logit_x0(pool, q)
        tau = float(np.quantile(df["volume_ref"].to_numpy(float), 0.75))
        print(f"\n[{cell}] n={len(df):,} tau={tau:.3f}", flush=True)

        # ---- (a) C++ full + rank sweep
        exp0 = gdir / f"verify_K{K}_tau0"
        export(gdir, pool.name, 0.0, 0, exp0)
        tf, inf_f = cpp_solve(exp0, "full", 0)
        of = float(inf_f["obj_feasible"])
        print(f"  C++ full: t={tf:6.1f}s obj={of:,.2f}", flush=True)
        rec(cell=cell, engine="cpp", what="full", r="", t_s=round(tf, 2), obj=of, speedup="")
        for rk in SOLVER_RANKS:
            expd = gdir / f"verify_K{K}_r{rk}"
            export(gdir, pool.name, tau, rk, expd)
            th, inf_h = cpp_solve(expd, "hard", rk)
            oh = float(inf_h["obj_feasible"])
            gap = 100 * (oh - min(oh, of)) / min(oh, of)
            print(f"  C++ hard r={rk:2d}: t={th:6.1f}s SPEEDUP={tf/th:5.2f}x "
                  f"gap={gap:+.4f}%", flush=True)
            rec(cell=cell, engine="cpp", what="hard", r=rk, t_s=round(th, 2), obj=oh,
                speedup=round(tf / th, 2), gap_pct=round(gap, 4))

        # ---- (b) Python full + hard (r=20), same instance/tau/tol
        P = ca.load_problem(str(gdir), pool.name, multi_path_only=True)
        t = time.perf_counter(); sf = ca.solve_full(P, tol=TOL, max_outer=40)
        tpf = time.perf_counter() - t
        xf = M.convert_euclid(sf["x_raw"], P["d"], P["p2od"])
        opf = float(P["bpr"].beckmann(M.link_flow(P, xf)))
        print(f"  PY  full: t={tpf:6.1f}s obj={opf:,.2f}", flush=True)
        rec(cell=cell, engine="py", what="full", r="", t_s=round(tpf, 2), obj=opf, speedup="")
        major = ca.split_major_minor(P, tau=tau)
        C = ca.build_compressed(P, major, 20)
        t = time.perf_counter()
        sc = ca.solve_compressed(P, C, regime="hard", tol=TOL, max_outer=40)
        tph = time.perf_counter() - t
        xc = M.convert_euclid(sc["x_raw"], P["d"], P["p2od"])
        oph = float(P["bpr"].beckmann(M.link_flow(P, xc)))
        gap = 100 * (oph - min(oph, opf)) / min(oph, opf)
        print(f"  PY  hard r=20: t={tph:6.1f}s SPEEDUP={tpf/tph:5.2f}x gap={gap:+.4f}%",
              flush=True)
        rec(cell=cell, engine="py", what="hard", r=20, t_s=round(tph, 2), obj=oph,
            speedup=round(tpf / tph, 2), gap_pct=round(gap, 4))

        # ---- (c) Python w0-free (consistent shift, mirrors the C++ blob fixer)
        C2 = dict(C)
        x0m = C["x0m"]
        B = P["B"].tocsr()
        minors = np.where(C["minor"])[0]
        vshift = np.zeros(B.shape[1])
        for k, p in enumerate(minors):
            if x0m[k] != 0.0:
                sl_ = slice(B.indptr[p], B.indptr[p + 1])
                vshift[B.indices[sl_]] += x0m[k]
        C2["v_base"] = C["v_base"] - vshift
        d_eff = C["d_eff"].copy()
        np.add.at(d_eff, P["p2od"][minors], x0m)
        C2["d_eff"] = d_eff
        C2["x0m"] = np.zeros_like(x0m)
        t = time.perf_counter()
        s0 = ca.solve_compressed(P, C2, regime="hard", tol=TOL, max_outer=40)
        tw = time.perf_counter() - t
        xw = M.convert_euclid(s0["x_raw"], P["d"], P["p2od"])
        ow = float(P["bpr"].beckmann(M.link_flow(P, xw)))
        gap = 100 * (ow - min(ow, opf)) / min(ow, opf)
        print(f"  PY  w0free r=20: t={tw:6.1f}s SPEEDUP={tpf/tw:5.2f}x gap={gap:+.4f}%",
              flush=True)
        rec(cell=cell, engine="py", what="w0free", r=20, t_s=round(tw, 2), obj=ow,
            speedup=round(tpf / tw, 2), gap_pct=round(gap, 4))

    print(f"\nwrote {out}\n[verify] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
