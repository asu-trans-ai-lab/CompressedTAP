"""Experiment 14 -- operator x compression at Regional X32 (Kbar ~ 30), plus latent atoms.

The constraints-vs-variables law was certified on the grid and Chicago Sketch: variable
reduction accelerates ALM (whose bottleneck is in the variables) and harms projection-type
operators (whose bottleneck is the per-OD structure that compression destroys). This runs
the SAME certified operator implementations, unmodified, on the Regional X32 instance
(top-20k ODs, ~30 paths/OD, n ~ 572k) -- the rich cell where ALM-compression pays -- and
adds the latent-atom (dynamic exchange) arm.

Arms (Python, single thread, matched tol 1e-4, per-arm wall caps):
  FULL        ALM (solve_full) | GP (op_gp_full, per-OD projection) | FW (op_fw_full)
  COMPRESSED  ALM-hard (weighted r=6, tau = 75th pct -- the exp13 configuration)
              GP-signed (certified impl; expected DNS: dense ell x ell projection with
                         ell = 19,284 ~ 3.0 GB and no separable structure)
              RG-anchor (certified impl)
  LATENT ATOM ALM on x = w0 + U_G z + A alpha: global r=6 backbone + B=2000 dynamically
              repriced, cost-selected, coefficient-bounded OD exchange atoms (exp11
              machinery), 8 outers
  FW compressed: NOT APPLICABLE, reported as a note -- the linear minimisation oracle
              over {A1 y + M z = d_eff, y >= 0, w0 + U z >= 0} is an LP, not a
              closed-form per-OD argmin; compression removes exactly the structure FW
              is fast because of.

Scoring: every terminal iterate goes through the same Euclidean per-OD conversion; the
common gap G (independent evaluator), feasible objective vs the best of all arms, in-pool
AON certificate, and per-family speedups t(full op) / t(compressed op).

    python drivers/exp14_operators_regional.py [--cap 1200] [--rank 6] [--B 2000]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))          # campaign config/common
sys.path.insert(0, str(HERE))                 # exp08/exp11 machinery
import common as K
import config as C

sys.path.insert(0, str(ROOT / "git_dev" / "CompressedTAP" / "repro_v3" / "python"))
import compressed_assignment as ca            # certified module (CERTPY on path via config)
import v3_metrics as M
from run_table2_consistency import LMOIndex, op_fw_full, op_gp_full
from gp_compressed import solve_gp_signed
from run_anchor_reduced import build_anchor, solve_R

from exp08_lockstep import CommonEvaluator
from exp11_hybrid import HybridStepper, build_atoms, od_gap_and_costs

DS = Path(os.environ.get("TMP", "/tmp")) / "extreme_case" / "regional" / "M20000"
POOL = "pool_X32.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--cap", type=float, default=1200.0)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--tau-q", type=float, default=0.75)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--atom-outers", type=int, default=8)
    a = ap.parse_args()
    rows = []
    K.banner(f"exp14 operators x compression, Regional X32 (r={a.rank}, B={a.B}, "
             f"cap {a.cap:.0f}s/arm)")

    P = ca.load_problem(str(DS), POOL, multi_path_only=True)
    tau = float(np.quantile(P["x0"], a.tau_q))
    major = ca.split_major_minor(P, tau=tau)
    Cb = ca.build_compressed(P, major, a.rank, weight_flows=P["x0"])
    red = 100 * (P["n"] - int(major.sum()) - Cb["r"]) / P["n"]
    print(f"  n={P['n']:,} od={P['n_od']:,} links={P['m']:,} tau={tau:.4f} "
          f"majors={int(major.sum()):,} r={Cb['r']} reduction={red:.1f}%", flush=True)
    ev = CommonEvaluator(P)
    lmo = LMOIndex(P["p2od"], P["n_od"], P["n"])
    v0 = P.get("v0", 0.0)

    def relgap(x):
        v = v0 + np.asarray(P["B"].T @ x).flatten()
        c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
        sx = lmo(c, P["d"])
        return float(c @ (x - sx)) / max(abs(float(c @ x)), 1e-12)

    def record(name, space, x_raw, secs, note=""):
        xf = M.convert_euclid(np.asarray(x_raw, float), P["d"], P["p2od"])
        vf = M.link_flow(P, xf)
        obj = float(P["bpr"].beckmann(vf))
        G = ev(xf)["G"]
        rows.append(dict(op=name, space=space, time_s=round(secs, 1),
                         obj_feasible=obj, G=round(G, 6),
                         cert_relgap=relgap(xf), note=note))
        print(f"  {name:11s} [{space:10s}] t={secs:8.1f}s obj={obj:,.2f} "
              f"G={G:.5f} {note}", flush=True)
        K.write("exp14_operators_regional_X32", rows)

    # ------------------------------------------------------------- full space
    print("[exp14] full space", flush=True)
    t = time.perf_counter()
    sf = ca.solve_full(P, tol=a.tol, max_outer=40, max_seconds=a.cap * 1.5)
    record("ALM", "full", sf["x_raw"], time.perf_counter() - t)

    xg, tg, itg, gg = op_gp_full(P, a.tol, max_seconds=a.cap)
    record("GP", "full", xg, tg, f"it={itg} gap={gg:.1e}")

    xw, tw, itw, gw = op_fw_full(P, a.tol, max_seconds=a.cap)
    record("FW", "full", xw, tw, f"it={itw} gap={gw:.1e}")

    # ------------------------------------------------------------- compressed
    print("[exp14] compressed (incumbent, weighted r=%d)" % a.rank, flush=True)
    t = time.perf_counter()
    sc = ca.solve_compressed(P, Cb, regime="hard", tol=a.tol, max_outer=30,
                             max_seconds=a.cap * 1.5)
    record("ALM-hard", "compressed", sc["x_raw"], time.perf_counter() - t)

    def cert_fn(x_full):
        return relgap(np.maximum(x_full, 0.0))
    t = time.perf_counter()
    try:
        gp_out = solve_gp_signed(P, Cb, cert_fn, tol=a.tol, max_seconds=a.cap)
        xr = gp_out[0] if isinstance(gp_out, tuple) else gp_out
        record("GP-signed", "compressed", xr, time.perf_counter() - t)
    except MemoryError:
        secs = time.perf_counter() - t
        rows.append(dict(op="GP-signed", space="compressed", time_s=round(secs, 1),
                         note=f"DNS MemoryError: dense ell x ell with ell={P['n_od']:,}"))
        print(f"  GP-signed   DNS (MemoryError, ell={P['n_od']:,})", flush=True)
        K.write("exp14_operators_regional_X32", rows)

    t = time.perf_counter()
    try:
        R = build_anchor(P, Cb)
        rr = solve_R(P, Cb, R, tol=a.tol, max_outer=30)
        record("RG-anchor", "compressed", rr["x"], time.perf_counter() - t,
               f"anchors_active={rr.get('anchor_active')}")
    except (MemoryError, Exception) as e:
        secs = time.perf_counter() - t
        rows.append(dict(op="RG-anchor", space="compressed", time_s=round(secs, 1),
                         note=f"DNS {type(e).__name__}: {str(e)[:120]}"))
        print(f"  RG-anchor   DNS ({type(e).__name__}: {str(e)[:120]})", flush=True)
        K.write("exp14_operators_regional_X32", rows)

    # ------------------------------------------------------------ latent atoms
    print("[exp14] latent atoms (dynamic exchange, B=%d)" % a.B, flush=True)
    x_start = ca.project_feasible(np.maximum(P["x0"], 1e-3), P["A"], P["d"], P["p2od"])
    minor_idx = np.where(Cb["minor"])[0]
    B2 = P["B"][Cb["minor"]]
    A2 = P["A"][:, Cb["minor"]]
    od_of_minor = P["p2od"][minor_idx]
    x_now = x_start.copy()
    lam = np.zeros(P["n_od"])
    rho, prev = 100.0, np.inf
    t_atom = 0.0
    for k in range(1, a.atom_outers + 1):
        x0m = np.maximum(x_now, 0.0)[minor_idx]
        d_eff = P["d"] - np.asarray(A2 @ x0m).flatten()
        v_base = v0 + np.asarray(B2.T @ x0m).flatten()
        Gq, c = od_gap_and_costs(P, x_now)
        active = np.argsort(Gq)[::-1][:a.B]
        A_at, bnd = build_atoms(od_of_minor, x0m, c[minor_idx], active, 1)
        U = sp.hstack([sp.csr_matrix(Cb["U"]), A_at]).tocsr()
        Cb2 = dict(B1=Cb["B1"], A1=Cb["A1"], U=U, D=(B2.T @ U), M=(A2 @ U),
                   d_eff=d_eff, v_base=v_base, x0m=x0m, major=Cb["major"],
                   minor=Cb["minor"], r=U.shape[1], svd_time=0.0)
        st = HybridStepper(P, Cb2, np.maximum(x_now, 0.0), a.tol, 200,
                           [(None, None)] * Cb["r"] + bnd)
        st.lam, st.rho, st.prev = lam.copy(), rho, prev
        rep = st.step()
        lam, rho, prev = st.lam.copy(), st.rho, st.prev
        x_now = st.export_x()
        t_atom += rep["cpu_s"]
        print(f"    atom outer {k}: G={ev(np.maximum(x_now,0))['G']:.5f} "
              f"cpu={rep['cpu_s']:.1f}s atoms={A_at.shape[1]:,}", flush=True)
    record("ALM-atoms", "compressed", x_now, t_atom, f"B={a.B}, dyn repriced")

    # ---------------------------------------------------------------- summary
    f_best = min(r["obj_feasible"] for r in rows if "obj_feasible" in r)
    tfull = {r["op"]: r["time_s"] for r in rows if r.get("space") == "full"}
    fam = {"ALM-hard": "ALM", "ALM-atoms": "ALM", "GP-signed": "GP", "RG-anchor": "GP"}
    print(f"\n[exp14] SUMMARY (f_best={f_best:,.2f}; FW compressed n/a: LMO over the "
          f"compressed polytope is an LP, not a closed-form oracle)")
    for r in rows:
        if "obj_feasible" in r:
            r["gap_vs_best_pct"] = round(100 * (r["obj_feasible"] - f_best) / f_best, 4)
            b = fam.get(r["op"])
            r["speedup_vs_full"] = (round(tfull[b] / r["time_s"], 2)
                                    if r.get("space") == "compressed" and b in tfull
                                    else "")
        print(f"  {r['op']:11s} {r.get('space',''):10s} t={r['time_s']:>8}s "
              f"gap%={r.get('gap_vs_best_pct','')!s:>8} G={r.get('G','')!s:>8} "
              f"S={r.get('speedup_vs_full','')!s:>6} {r.get('note','')}")
    K.write("exp14_operators_regional_X32", rows)
    print(f"wrote {C.RESULTS/'exp14_operators_regional_X32.csv'}\n[exp14] DONE",
          flush=True)


if __name__ == "__main__":
    main()
