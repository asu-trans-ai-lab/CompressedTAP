"""Operator x compression comparison on Chicago Sketch (author's P2, 2026-07-24).

One instance: the submitted V2 Sketch pool (n=42,774), tau=4.54 (53.1% reduction -- the
threshold where compression wins), r=50, matched working tolerance 1e-4, single thread.

Rows:
  full space   : ALM (solve_full), GP (op_gp_full; classical per-OD-projection GP IS the
                 full-space reduced-gradient family), FW (op_fw_full)
  compressed   : ALM-hard (solve_compressed), GP-signed (solve_gp_signed),
                 RG-anchor (solve_R: anchor substitution, conservation exact by construction)
  FW compressed: not applicable -- the LMO over {A1 y + M z = d_eff, y>=0, w0+U z>=0} is an
                 LP, not a closed-form oracle. Reported as a note, not a number.

Headline speedups (full / compressed, per operator family):
  ALM   : t(ALM-full)  / t(ALM-hard)
  GP    : t(GP-full)   / t(GP-signed)
  RG    : t(GP-full)   / t(RG-anchor)   [full-space RG == per-OD-projection GP]

All terminal iterates go through the SAME v3_metrics Euclidean conversion; feasible objective
gaps are vs the best feasible point found by ANY of the six runs (decision A).

    python run_sketch_operators.py [--tau 4.54] [--rank 50] [--tol 1e-4] [--cap 900]
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
from run_table2_consistency import op_gp_full, op_fw_full, LMOIndex
from gp_compressed import solve_gp_signed
from run_anchor_reduced import build_anchor, solve_R

SKETCH = ROOT / "OR_paper_revision_V2" / "m4_rerun" / "data" / "sketch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=4.54)
    ap.add_argument("--rank", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--cap", type=float, default=900.0, help="wall cap per slow operator (s)")
    a = ap.parse_args()

    print(f"[ops] loading sketch (V2 pool)", flush=True)
    P = ca.load_problem(str(SKETCH), "pool.csv", multi_path_only=True)
    major = ca.split_major_minor(P, tau=a.tau)
    C = ca.build_compressed(P, major, a.rank)
    s = int(major.sum())
    red = 100 * (P["n"] - s - C["r"]) / P["n"]
    print(f"  n={P['n']:,} od={P['n_od']:,} tau={a.tau} majors={s:,} r={C['r']} "
          f"({red:.1f}% reduction)", flush=True)
    sl = M.od_slice_index(P)

    lmo = LMOIndex(P["p2od"], P["n_od"], P["n"])
    v0 = P.get("v0", 0.0)

    def relgap(x):
        v = v0 + np.asarray(P["B"].T @ x).flatten()
        c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
        sx = lmo(c, P["d"])
        return float(c @ (x - sx)) / max(abs(float(c @ x)), 1e-12)

    runs = []

    def record(name, space, x_raw, secs, note=""):
        xf = M.convert_euclid(np.asarray(x_raw, float), P["d"], P["p2od"])
        vf = M.link_flow(P, xf)
        obj = float(P["bpr"].beckmann(vf))
        cert = relgap(xf)
        runs.append(dict(op=name, space=space, time_s=round(secs, 2), obj_feasible=obj,
                         cert_relgap=cert, note=note))
        print(f"  {name:10s} [{space:10s}] t={secs:8.2f}s obj={obj:,.2f} "
              f"cert={cert:.3e} {note}", flush=True)

    # ---------- full space ----------
    print("[ops] full space", flush=True)
    t = time.perf_counter(); sf = ca.solve_full(P, tol=a.tol, max_outer=40)
    record("ALM", "full", sf["x_raw"], time.perf_counter() - t)

    xg, tg, itg, gg = op_gp_full(P, a.tol, max_seconds=a.cap)
    record("GP", "full", xg, tg, f"it={itg} gap={gg:.1e}")
    t_gp_full = tg

    xw, tw, itw, gw = op_fw_full(P, a.tol, max_seconds=a.cap)
    record("FW", "full", xw, tw, f"it={itw} gap={gw:.1e}")

    # ---------- compressed (same signed-SVD C) ----------
    print("[ops] compressed (tau=%.2f, r=%d)" % (a.tau, a.rank), flush=True)
    t = time.perf_counter(); sc = ca.solve_compressed(P, C, regime="hard", tol=a.tol,
                                                      max_outer=30)
    record("ALM", "compressed", sc["x_raw"], time.perf_counter() - t)

    def cert_fn(x_full):
        return relgap(np.maximum(x_full, 0.0))
    t = time.perf_counter()
    gp_out = solve_gp_signed(P, C, cert_fn, tol=a.tol, max_seconds=a.cap)
    xr = gp_out[0] if isinstance(gp_out, tuple) else gp_out
    record("GP-signed", "compressed", xr, time.perf_counter() - t,
           "" if not isinstance(gp_out, tuple) else f"cert={gp_out[-1]:.1e}")

    R = build_anchor(P, C)
    t = time.perf_counter(); rr = solve_R(P, C, R, tol=a.tol, max_outer=30)
    record("RG-anchor", "compressed", rr["x"], time.perf_counter() - t,
           f"anchors_active={rr['anchor_active']}")

    # ---------- summary ----------
    f_best = min(r["obj_feasible"] for r in runs)
    tfull = {r["op"]: r["time_s"] for r in runs if r["space"] == "full"}
    for r in runs:
        r["gap_vs_best_pct"] = 100 * (r["obj_feasible"] - f_best) / f_best
        base = dict(ALM="ALM", **{"GP-signed": "GP", "RG-anchor": "GP"}).get(r["op"])
        r["speedup_vs_full"] = (round(tfull[base] / r["time_s"], 2)
                                if r["space"] == "compressed" and base in tfull else "")

    out = HERE.parent / "results" / "sketch_operators.csv"
    keys = ["op", "space", "time_s", "speedup_vs_full", "obj_feasible",
            "gap_vs_best_pct", "cert_relgap", "note"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"\n[ops] SUMMARY (f_best={f_best:,.2f})")
    print(f"  {'op':10s} {'space':10s} {'t(s)':>8} {'speedup':>8} {'gap%':>9} {'cert':>10}")
    for r in runs:
        print(f"  {r['op']:10s} {r['space']:10s} {r['time_s']:>8.2f} "
              f"{str(r['speedup_vs_full']):>8} {r['gap_vs_best_pct']:>9.4f} "
              f"{r['cert_relgap']:>10.2e}")
    print(f"  (FW compressed: n/a -- LMO over the compressed polytope is an LP, "
          f"not a closed-form oracle)")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
