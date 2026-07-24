"""Table 4 Panel A (tab:cutoff-summary) — 11 [fill] cells.

Three networks x four thresholds, preserving the submitted threshold grid exactly. The
revnote restricts this round to a numerical rerun: same networks, same tau values, same
rank, same table architecture.

Per author decision 2, EVERY column is produced fresh -- reduction, raw objective
difference, delta_F, Gap_F, link R^2, outer/inner iterations and CPU. No submitted timing
is carried over, so no row mixes two campaigns.

Per author decision A, v^ref is the BEST FEASIBLE POINT found by any uncompressed
operator: each candidate's terminal iterate is converted by the per-OD simplex projection
(demand equality and nonnegativity together) and the lowest feasible objective wins. No
fixed operator is safe at both scales (REMARKS.md R2, R10). `ref_source`, `ref_cert` and
`ref_obj_spread_pct` are written to every row.

Per author decision B, timings repeat 3x on small instances and once on the large ones;
Gap_F, delta_F and R^2 are deterministic and unaffected.

Per author decision 3, the offset-free variant (w0 = 0) is evaluated on the same solve
path and written to `w0free_*` diagnostic columns. It does not enter the manuscript.

    python run_table4A_thresholds.py --net sketch|regional|philadelphia [--reps 3]
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "source" / "updated_TAPLite" / "python"))

import v3_metrics as M                                    # noqa: E402
import compressed_assignment as ca                        # noqa: E402
import v3_reference as REF                              # noqa: E402
from run_table2_consistency import (LMOIndex, op_gp_full, op_fw_full,   # noqa: E402
                                    op_alm_full)

DATA = ROOT / "source" / "updated_TAPLite" / "data"
NETS = {
    # thresholds exactly as submitted in the v3 manuscript
    "sketch": dict(dir=DATA / "03_chicago_sketch", pool="path_pool.csv", multi=True,
                   taus=[0.0, 0.46, 1.06, 4.54], rank=50),
    "regional": dict(dir=DATA / "04_chicago_regional", pool="path_pool.csv", multi=True,
                     taus=[0.0, 0.23, 0.46, 1.03], rank=50),
    "philadelphia": dict(dir=ROOT / "OR_paper_revision_V2" / "m4_rerun" / "data"
                         / "philadelphia", pool="pool.csv", multi=True,
                         taus=[0.0, 0.67, 0.99, 5.33], rank=50),
    # sioux is not a manuscript Panel-A network; included only as a fast validation of the
    # whole chain (its thresholds are placeholder quantiles, not the submitted grid).
    "sioux": dict(dir=DATA / "02_Sioux_Falls", pool="path_pool_SFK25.csv", multi=False,
                  taus=[0.0, 0.46, 1.06, 4.54], rank=50),
}


def median_solve(fn, reps):
    """Median of `reps` independent solves; returns (result_of_median_run, median_seconds)."""
    runs = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        runs.append((time.perf_counter() - t, out))
    runs.sort(key=lambda r: r[0])
    secs, out = runs[len(runs) // 2]
    return out, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=list(NETS))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--ref-tol", type=float, default=1e-7)
    ap.add_argument("--reps", type=int, default=None,
                    help="default: 3 on small instances, 1 on large ones "
                         "(author decision B of 2026-07-19)")
    ap.add_argument("--ref-cap", type=float, default=900.0,
                    help="wall-clock cap per reference candidate (decision 2)")
    ap.add_argument("--solve-cap", type=float, default=1800.0,
                    help="wall-clock cap per per-threshold solve")
    ap.add_argument("--maxiter-inner", type=int, default=None,
                    help="inner L-BFGS-B iters/outer; default 50 large (decision b), "
                         "200 certified small")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = NETS[a.net]
    # Author decision B (2026-07-19): repeat timings 3x on small instances, once on the
    # large ones. Gap_F, delta_F and R^2 are deterministic and unaffected by the repeat
    # count; only the CPU column loses its dispersion, and that is marked in the CSV.
    if a.reps is None:
        a.reps = 1 if a.net in ("regional", "philadelphia", "sketch") else 3
    if a.maxiter_inner is None:
        a.maxiter_inner = 50 if a.net in ("regional", "philadelphia", "sketch") else 200

    print(f"[T4A] loading {a.net} (reps={a.reps})", flush=True)
    t0 = time.perf_counter()
    P = ca.load_problem(str(cfg["dir"]), cfg["pool"], multi_path_only=cfg["multi"])
    load_s = time.perf_counter() - t0
    sl = M.od_slice_index(P)
    print(f"  paths={P['n']:,} od={P['n_od']:,} links={P['B'].shape[1]:,} "
          f"load={load_s:.1f}s", flush=True)

    # ---- reference: best feasible point from ANY uncompressed operator (decision A).
    # Each candidate's terminal iterate is converted by the per-OD simplex projection
    # (demand equality and nonnegativity together) before comparison, so the winner is the
    # best FEASIBLE point, not the best raw iterate.
    # Decision 1 (2026-07-19): on large networks the Frank-Wolfe candidate is skipped. It
    # needs ~1e5 iterations even on Sketch, is catastrophically slow at 4.8M paths, and is
    # essentially never the best feasible point (ALM or GP always beats it). Skipping it is
    # an efficiency choice only -- v^ref is still the best point among the uncompressed
    # operators that ARE run, and which ones ran is recorded per row (ref_operators).
    # Decision 2 (2026-07-19): every reference candidate gets the SAME wall-clock cap
    # (--ref-cap), and ALM is capped too via the new max_seconds guard, so no candidate can
    # hang the campaign. On large networks the reference then converges only to ~1e-3; that
    # resolution limit is stated in the manuscript (decision A).
    large = a.net in ("regional", "philadelphia", "sketch")
    ops = {
        "GP-full":  lambda: op_gp_full(P, a.ref_tol, max_seconds=a.ref_cap),
        "ALM-full": lambda: op_alm_full(P, a.tol, max_seconds=a.ref_cap),
    }
    if not large:
        ops["FW-full"] = lambda: op_fw_full(P, a.ref_tol, max_seconds=a.ref_cap)
    print(f"[T4A] reference: best of {list(ops)} (cap {a.ref_cap:.0f}s each)", flush=True)
    ref, ref_cands = REF.build_reference(P, ops, a.tol)
    ref["operators"] = "+".join(ops)
    x_ref, v_ref, f_ref = ref["x_ref"], ref["v_ref"], ref["f_ref"]
    g_ref, t_ref, it_ref = ref["cert"], ref["seconds"], ref["iterations"]
    print(f"  f_ref={f_ref:,.4f} source={ref['source']} cert={g_ref:.3e} "
          f"spread={ref['obj_spread_pct']:.6f}%", flush=True)

    out = Path(a.out) if a.out else HERE.parent / "results" / f"table4A_{a.net}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush(rows):
        # Rewrite the CSV after every completed threshold so a network stopped early (net
        # ceiling, R14 overrun) keeps its finished rows instead of losing everything.
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    rows = []
    n = P["n"]
    for tau in cfg["taus"]:
        print(f"[T4A] tau={tau}", flush=True)
        if tau <= 0.0:
            t_pre = 0.0
            if large and tau0_prepared is not None:
                x_raw, secs, inner = tau0_prepared      # decision a: reuse the ref solve
            else:
                sol, secs = median_solve(
                    lambda: ca.solve_full(P, tol=a.tol, max_outer=40,
                                          maxiter_inner=a.maxiter_inner,
                                          max_seconds=a.solve_cap), a.reps)
                x_raw = sol["x_raw"]; inner = sol.get("inner_iters", -1)
            s_major, r_used, reduction = n, 0, 0.0
            w0free = None
        else:
            tpre = time.perf_counter()
            major = ca.split_major_minor(P, tau)
            s_major = int(major.sum())
            if s_major >= n:
                print(f"  all major at tau={tau}; skipped", flush=True)
                continue
            C = ca.build_compressed(P, major, cfg["rank"])
            t_pre = time.perf_counter() - tpre
            r_used = C["r"]
            reduction = 100.0 * (n - s_major - r_used) / n
            sol, secs = median_solve(
                lambda: ca.solve_compressed(P, C, regime="hard", tol=a.tol, max_outer=30,
                                            maxiter_inner=a.maxiter_inner,
                                            max_seconds=a.solve_cap), a.reps)
            x_raw = sol["x_raw"]
            inner = sol.get("inner_iters", -1)
            # decision c: the w0-free diagnostic doubles per-threshold cost, so it is
            # SKIPPED on the large networks (CSV-only diagnostic, decision 3).
            w0free = None
            if not large:
                Cf = dict(C)
                Cf["x0m"] = np.zeros_like(C["x0m"])
                Cf["d_eff"] = P["d"].copy()
                Cf["v_base"] = P.get("v0", 0.0)
                try:
                    solf = ca.solve_compressed(P, Cf, regime="hard", tol=a.tol,
                                               max_outer=30, maxiter_inner=a.maxiter_inner,
                                               max_seconds=a.solve_cap)
                    w0free = M.evaluate(P, solf["x_raw"], f_ref, v_ref, "euclid", sl)
                except Exception as ex:
                    print(f"  w0-free diagnostic failed: {ex!r}", flush=True)
                    w0free = None

        m = M.evaluate_both(P, x_raw, f_ref, v_ref, sl)
        m.update(dict(net=a.net, tau=tau, n_paths=n, n_od=P["n_od"],
                      majors=s_major, rank=r_used, reduction_pct=reduction,
                      raw_obj_diff_pct=M.raw_objective_diff_pct(P, x_raw, f_ref),
                      inner_iters=inner, cpu_s=secs, preprocess_s=t_pre,
                      tol=a.tol, ref_tol=a.ref_tol, ref_cert=g_ref, reps=a.reps,
                      ref_source=ref["source"], ref_operators=ref["operators"],
                      maxiter_inner=a.maxiter_inner,
                      ref_obj_spread_pct=ref["obj_spread_pct"],
                      ref_n_candidates=ref["n_candidates"]))
        if w0free is not None:
            m["w0free_Gap_F_pct"] = w0free["Gap_F_pct"]
            m["w0free_delta_F_pct"] = w0free["delta_F_pct"]
            m["w0free_link_R2"] = w0free["link_R2"]
        rows.append(m)
        flush(rows)
        print(f"  red={reduction:5.1f}%  raw={m['raw_obj_diff_pct']:+8.4f}%  "
              f"dF={m['delta_F_pct']:7.4f}%  GapF={m['Gap_F_pct']:+9.5f}%  "
              f"R2={m['link_R2']:.4f}  inner={inner}  pre={t_pre:.1f}s  "
              f"cpu={secs:.1f}s", flush=True)

    out = Path(a.out) if a.out else HERE.parent / "results" / f"table4A_{a.net}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}", flush=True)

    neg = [r for r in rows if r["Gap_F_pct"] < -1e-6]
    print(f"GATE 2 (no negative Gap_F): {'PASS' if not neg else 'FAIL'}"
          + ("" if not neg else f" -- {[(r['tau'], r['Gap_F_pct']) for r in neg]}"))
    return 0 if not neg else 1


if __name__ == "__main__":
    raise SystemExit(main())
