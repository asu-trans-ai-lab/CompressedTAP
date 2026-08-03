"""Benefit of compression per OPERATOR on the grid (author request 2026-07-25), w0 included.

Cells: N in {16,32} x K in {16,32,64} (the home-regime cells). Same fixed rules as the grid
matrix: left-col->right-col demand q=720/N, logit x0, tau = 75th pct of x0, r=20.

Operators (Python, single engine so all four exist; within-engine speedups):
  full space  : ALM (solve_full), GP (op_gp_full), FW (op_fw_full)
                [full-space reduced gradient == per-OD-projection GP]
  compressed  : ALM-hard (solve_compressed), GP-signed (solve_gp_signed -- feasible here:
                ell <= 1024 so the exact projection's dense ell x ell factorization is small),
                RG-anchor (solve_R, same signed w0 basis)
  FW compressed: n/a (the LMO over {A1 y + M z = d_eff, y>=0, w0+Uz>=0} is an LP).

Speedup columns: ALM t_full/t_hard; GP t_gpfull/t_gpsigned; RG t_gpfull/t_rganchor.
All terminal iterates -> the shared v3_metrics Euclidean conversion; gaps vs the best
feasible point of the cell. C++ ALM cross-check lives in grid_axis_cpp.csv.

    python run_grid_operators.py [--cells 16:16,16:32,16:64,32:16,32:32,32:64]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CERTPY = ROOT / "vendor"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CERTPY))

import v3_metrics as M
import compressed_assignment as ca
from run_table2_consistency import op_gp_full, op_fw_full, LMOIndex
from gp_compressed import solve_gp_signed
from run_anchor_reduced import build_anchor, solve_R
from run_grid_axis_cpp import gen_grid, logit_x0, KSP, ENV

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "grid_axis"
RANK = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="16:16,16:32,16:64,32:16,32:32,32:64")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--cap", type=float, default=600.0)
    a = ap.parse_args()
    out = HERE.parent / "results" / "grid_operators.csv"
    rows = []

    def flush():
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    for cell in [c.strip() for c in a.cells.split(",")]:
        N, K = (int(x) for x in cell.split(":"))
        gdir = SCRATCH / f"g{N}"
        m, n_od, q = gen_grid(N, gdir)
        pool = gdir / f"ops_pool_K{K}.csv"
        r = subprocess.run([str(KSP), str(gdir), str(K), "1.4", str(pool)],
                           capture_output=True, text=True, env=ENV)
        if r.returncode != 0:
            print(f"[{cell}] ksp_gen failed: {r.stderr[-200:]}"); continue
        logit_x0(pool, q)

        P = ca.load_problem(str(gdir), pool.name, multi_path_only=True)
        tau = float(np.quantile(P["x0"], 0.75))
        major = ca.split_major_minor(P, tau=tau)
        C = ca.build_compressed(P, major, RANK)
        s = int(major.sum())
        red = 100 * (P["n"] - s - C["r"]) / P["n"]
        print(f"\n[{cell}] n={P['n']:,} ell={P['n_od']} red={red:.1f}% r={C['r']}", flush=True)
        sl = M.od_slice_index(P)
        lmo = LMOIndex(P["p2od"], P["n_od"], P["n"])
        v0 = P.get("v0", 0.0)

        def relgap(x):
            v = v0 + np.asarray(P["B"].T @ x).flatten()
            c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
            return float(c @ (x - lmo(c, P["d"]))) / max(abs(float(c @ x)), 1e-12)

        res = {}

        def record(op, space, x_raw, secs, note=""):
            xf = M.convert_euclid(np.asarray(x_raw, float), P["d"], P["p2od"])
            obj = float(P["bpr"].beckmann(M.link_flow(P, xf)))
            res[(op, space)] = dict(t=secs, obj=obj, cert=relgap(xf), note=note)
            print(f"  {op:10s}[{space:4s}] t={secs:7.2f}s obj={obj:,.2f} "
                  f"cert={res[(op,space)]['cert']:.2e} {note}", flush=True)

        t = time.perf_counter(); sf = ca.solve_full(P, tol=a.tol, max_outer=40)
        record("ALM", "full", sf["x_raw"], time.perf_counter() - t)
        xg, tg, itg, gg = op_gp_full(P, a.tol, max_seconds=a.cap)
        record("GP", "full", xg, tg, f"it={itg}")
        xw, tw, itw, gw = op_fw_full(P, a.tol, max_seconds=a.cap)
        record("FW", "full", xw, tw, f"it={itw}")

        t = time.perf_counter()
        sc = ca.solve_compressed(P, C, regime="hard", tol=a.tol, max_outer=30)
        record("ALM", "comp", sc["x_raw"], time.perf_counter() - t)

        def cert_fn(x_full):
            return relgap(np.maximum(x_full, 0.0))
        t = time.perf_counter()
        gp_out = solve_gp_signed(P, C, cert_fn, tol=a.tol, max_seconds=a.cap)
        xr = gp_out[0] if isinstance(gp_out, tuple) else gp_out
        record("GP-signed", "comp", xr, time.perf_counter() - t)

        R = build_anchor(P, C)
        t = time.perf_counter(); rr = solve_R(P, C, R, tol=a.tol, max_outer=30)
        record("RG-anchor", "comp", rr["x"], time.perf_counter() - t,
               f"anchors={rr['anchor_active']}")

        f_best = min(v["obj"] for v in res.values())
        su = {
            "ALM": res[("ALM", "full")]["t"] / res[("ALM", "comp")]["t"],
            "GP": res[("GP", "full")]["t"] / res[("GP-signed", "comp")]["t"],
            "RG": res[("GP", "full")]["t"] / res[("RG-anchor", "comp")]["t"],
        }
        print(f"  --> SPEEDUPS  ALM {su['ALM']:.2f}x   GP {su['GP']:.2f}x   "
              f"RG {su['RG']:.2f}x   (FW-full baseline {res[('FW','full')]['t']:.2f}s)",
              flush=True)
        for (op, space), v in res.items():
            rows.append(dict(N=N, K=K, n=P["n"], ell=P["n_od"],
                             reduction_pct=round(red, 1), op=op, space=space,
                             t_s=round(v["t"], 2),
                             speedup=(round(su.get(op if op in ("ALM",) else
                                                   {"GP-signed": "GP",
                                                    "RG-anchor": "RG"}.get(op, ""), 0), 2)
                                      if space == "comp" else ""),
                             obj=v["obj"],
                             gap_vs_best_pct=round(100 * (v["obj"] - f_best) / f_best, 4),
                             cert=f"{v['cert']:.2e}", note=v["note"]))
        flush()

    print("\n[grid-ops] SUMMARY (compression speedup per operator):")
    print(f"  {'cell':8s} {'ALM':>7} {'GP':>7} {'RG':>7}")
    cells = sorted({(r_["N"], r_["K"]) for r_ in rows})
    for (N, K) in cells:
        g = {r_["op"]: r_ for r_ in rows if r_["N"] == N and r_["K"] == K
             and r_["space"] == "comp"}
        print(f"  N{N}K{K:<4} {g['ALM']['speedup']:>7} {g['GP-signed']['speedup']:>7} "
              f"{g['RG-anchor']['speedup']:>7}")
    print(f"wrote {out}\n[grid-ops] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
