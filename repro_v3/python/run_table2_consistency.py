"""Table 2 (tab:operator-consistency) — 23 [fill] cells.

Verifies that the AL implementation does not alter the underlying Beckmann problem and
that the compressed formulation is not tied to the AL operator, under matched stopping
tolerances and the common v3 feasibility conversion.

Rows
    Full (tau=0)      ALM  |  Frank-Wolfe  |  Gradient projection
    Existing (tau,r)  ALM  |  GP-signed / GP-grouped   (decision 1: both, labelled)

Columns (per manuscript Eq. (metric))
    raw obj. diff. (%)    diagnostic only, evaluated at the UNPROJECTED terminal x~
    ||A x~ - d||_inf      residual before conversion
    delta_F (%)           size of the common feasibility conversion
    feasible obj. gap (%) the substantive measure, Gap_F
    link-flow diff. (%)

The reference v^ref / f^ref is the consistently converged tau=0 ALM run, evaluated
through v3_metrics with the Euclidean conversion (spec/TABLES.md).

    python run_table2_consistency.py --net sketch [--tol 1e-6] [--reps 3]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CERT = HERE.parents[1] / "vendor"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CERT))

import v3_metrics as M                                    # noqa: E402
import compressed_assignment as ca                        # noqa: E402
import gp_compressed as GPC                                # noqa: E402

DATA = HERE.parents[1] / "data"
NETS = {
    "sketch": dict(dir=DATA / "chicago_sketch", pool="path_pool.csv", multi=True,
                   tau_q=0.90, rank=50),
    "sioux": dict(dir=DATA / "02_Sioux_Falls", pool="path_pool_SFK25.csv", multi=False,
                  tau_q=0.90, rank=50),
}


# ------------------------------------------------------------------ operators
def op_alm_full(P, tol, max_outer=40, max_seconds=None):
    t = time.perf_counter()
    s = ca.solve_full(P, tol=tol, max_outer=max_outer, max_seconds=max_seconds)
    return s["x_raw"], time.perf_counter() - t, s.get("inner_iters", -1), float("nan")


class LMOIndex:
    """Linear minimization oracle on the OD-simplex product: all demand on the cheapest
    path of each OD pair.

    The grouping by OD does not change between iterations, so the sort that makes each
    OD's paths contiguous is done ONCE here; each call is then two C-level segmented
    reductions. The per-OD Python loop this replaces ran once per FW/GP iteration and
    cost 82,083 interpreter iterations on Chicago Sketch alone. (A per-call lexsort was
    tried first and is slower than the loop on small instances -- see REMARKS.md R9.)
    """

    def __init__(self, p2od, n_od, n):
        self.n = n
        self.order = np.argsort(p2od, kind="stable")
        ods = p2od[self.order]
        starts = np.searchsorted(ods, np.arange(n_od))
        sizes = np.diff(np.append(starts, n))
        self.nz = sizes > 0
        self.starts = starts[self.nz]
        self.sizes = sizes[self.nz]
        self.pos = np.arange(n)

    def __call__(self, c, d):
        s = np.zeros(self.n)
        cs = c[self.order]
        blockmin = np.minimum.reduceat(cs, self.starts)
        hit = cs == np.repeat(blockmin, self.sizes)
        first = np.minimum.reduceat(np.where(hit, self.pos, self.n), self.starts)
        s[self.order[first]] = d[self.nz]
        return s


def op_fw_full(P, tol, max_iter=200000, max_seconds=600.0):
    """Frank-Wolfe on the full path polytope.

    The line search works in LINK space: with dv = B'(s - x) precomputed once per
    iteration, v(alpha) = v + alpha*dv costs O(m) per trial instead of a full path-link
    matvec.  phi'(alpha) = t(v + alpha*dv) . dv is increasing in alpha (monotone link
    costs), so the exact step is found by bisection on the derivative.
    """
    B, d, p2od, bpr = P["B"], P["d"], P["p2od"], P["bpr"]
    n = P["n"]
    lmo = LMOIndex(p2od, len(d), n)
    x = M.convert_euclid(np.maximum(P["x0"], 0.0), d, p2od)
    v0 = P.get("v0", 0.0)
    v = v0 + np.asarray(B.T @ x).flatten()
    t = time.perf_counter()
    it = 0
    gap_rel = np.inf
    best_x, best_gap = x.copy(), np.inf
    for it in range(1, max_iter + 1):
        c = np.asarray(B @ bpr.t(v)).flatten()
        s_lmo = lmo(c, d)
        dx = s_lmo - x
        gap_rel = float(c @ (x - s_lmo)) / max(abs(float(c @ x)), 1e-12)
        if gap_rel < best_gap:
            best_gap, best_x = gap_rel, x.copy()
        if gap_rel <= tol or (time.perf_counter() - t) > max_seconds:
            break
        dv = np.asarray(B.T @ dx).flatten()
        if float(bpr.t(v) @ dv) >= 0.0:          # no descent at alpha = 0
            break
        lo, hi = 0.0, 1.0
        if float(bpr.t(v + dv) @ dv) <= 0.0:     # full step still descending
            alpha = 1.0
        else:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if float(bpr.t(v + mid * dv) @ dv) > 0.0:
                    hi = mid
                else:
                    lo = mid
            alpha = 0.5 * (lo + hi)
        x = x + alpha * dx
        v = v + alpha * dv
    return best_x, time.perf_counter() - t, it, best_gap


def op_gp_full(P, tol, max_iter=200000, max_seconds=600.0):
    """Gradient projection on the full OD-simplex product (BB step).

    Stops on the SAME certificate as op_fw_full (relative FW duality gap), so the two
    operators are compared at matched accuracy rather than on their own private criteria.
    """
    B, d, p2od, bpr = P["B"], P["d"], P["p2od"], P["bpr"]
    n = P["n"]
    lmo = LMOIndex(p2od, len(d), n)
    x = M.convert_euclid(np.maximum(P["x0"], 0.0), d, p2od)
    v0 = P.get("v0", 0.0)
    t = time.perf_counter()
    xp = gp = None
    it = 0
    gap_rel = np.inf
    # The Barzilai-Borwein step is deliberately NONMONOTONE: the objective may rise before
    # it falls, so the LAST iterate is not necessarily the best one found. Returning it
    # made a 1200 s reference run worse than a 600 s run of the same operator on Chicago
    # Sketch (REMARKS.md R11). Track and return the best iterate by certificate.
    best_x, best_gap = x.copy(), np.inf
    for it in range(1, max_iter + 1):
        v = v0 + np.asarray(B.T @ x).flatten()
        c = np.asarray(B @ bpr.t(v)).flatten()
        s_lmo = lmo(c, d)
        gap_rel = float(c @ (x - s_lmo)) / max(abs(float(c @ x)), 1e-12)
        if gap_rel < best_gap:
            best_gap, best_x = gap_rel, x.copy()
        if gap_rel <= tol or (time.perf_counter() - t) > max_seconds:
            break
        if xp is None:
            step = 1.0 / max(float(np.max(np.abs(c))), 1e-12)
        else:
            s_, y_ = x - xp, c - gp
            sy = float(s_ @ y_)
            step = float(s_ @ s_) / sy if sy > 1e-30 else 1.0
            if not (1e-14 < step < 1e14):
                step = 1.0
        xp, gp = x, c
        x = M.convert_euclid(x - step * c, d, p2od)
    return best_x, time.perf_counter() - t, it, best_gap


def op_alm_compressed(P, C, tol, max_outer=30):
    t = time.perf_counter()
    s = ca.solve_compressed(P, C, regime="hard", tol=tol, max_outer=max_outer)
    return s["x_raw"], time.perf_counter() - t, s.get("inner_iters", -1)


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="sketch", choices=list(NETS))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--ref-tol", type=float, default=1e-9)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=None)
    a = ap.parse_args()
    cfg = dict(NETS[a.net])
    if a.rank is not None:
        cfg["rank"] = a.rank

    print(f"[table2] loading {a.net}", flush=True)
    P = ca.load_problem(str(cfg["dir"]), cfg["pool"], multi_path_only=cfg["multi"])
    sl = M.od_slice_index(P)
    print(f"  paths={P['n']} od={P['n_od']} links={P['B'].shape[1]}", flush=True)

    # ---- shared optimality certificate (relative FW duality gap of the converted point)
    lmo_ref = LMOIndex(P["p2od"], P["n_od"], P["n"])

    def cert(x_raw):
        xf = M.convert_euclid(x_raw, P["d"], P["p2od"], sl)
        v = M.link_flow(P, xf)
        c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
        sm = lmo_ref(c, P["d"])
        return float(c @ (xf - sm)) / max(abs(float(c @ xf)), 1e-12)

    # ---- reference: full-path gradient projection converged to ref_tol.
    # The tau=0 ALM run is NOT used as v^ref: it terminates ~0.048% above the optimum and
    # cannot be driven closer through its exposed tolerances (results/GATE2_REFERENCE_
    # FINDING.md), which would bias every Gap_F low and reintroduce negative gaps.
    print(f"[table2] reference: full-path GP to certificate {a.ref_tol}", flush=True)
    x_ref_raw, t_ref, it_ref, g_ref = op_gp_full(P, a.ref_tol, max_seconds=1200)
    x_ref = M.convert_euclid(x_ref_raw, P["d"], P["p2od"], sl)
    v_ref = M.link_flow(P, x_ref)
    f_ref = float(P["bpr"].beckmann(v_ref))
    print(f"  f_ref={f_ref:,.6f}  cert={g_ref:.3e}  t={t_ref:.2f}s  it={it_ref}",
          flush=True)

    rows = []

    def record(rep, method, x_raw, secs, iters, cert_gap=float("nan")):
        m = M.evaluate_both(P, x_raw, f_ref, v_ref, sl)
        m.update(dict(net=a.net, representation=rep, method=method,
                      raw_obj_diff_pct=M.raw_objective_diff_pct(P, x_raw, f_ref),
                      cpu_s=secs, iterations=iters, tol=a.tol,
                      achieved_cert_gap=cert_gap))
        rows.append(m)
        print(f"  {rep:16s} {method:22s} raw={m['raw_obj_diff_pct']:+8.4f}%  "
              f"res={m['demand_residual_inf']:.2e}  dF={m['delta_F_pct']:.4f}%  "
              f"GapF={m['Gap_F_pct']:+.6f}%  link={m['link_diff_pct']:.4f}%  "
              f"t={secs:.2f}s", flush=True)

    record("Full (tau=0)", "Gradient projection [reference]", x_ref_raw, t_ref,
           it_ref, g_ref)

    print("[table2] full ALM", flush=True)
    xa_, ta_, ita_, _ = op_alm_full(P, a.tol)
    record("Full (tau=0)", "ALM", xa_, ta_, ita_, cert(xa_))

    print("[table2] full Frank-Wolfe", flush=True)
    xf_, tf_, itf_, gf_ = op_fw_full(P, a.tol)
    record("Full (tau=0)", "Frank-Wolfe", xf_, tf_, itf_, gf_)

    print("[table2] full gradient projection", flush=True)
    xg_, tg_, itg_, gg_ = op_gp_full(P, a.tol)
    record("Full (tau=0)", "Gradient projection", xg_, tg_, itg_, gg_)

    # ---- compressed at the existing (tau, r)
    x0pos = P["x0"][P["x0"] > 0]
    tau = float(np.quantile(x0pos, cfg["tau_q"]))
    major = ca.split_major_minor(P, tau)
    print(f"[table2] compressed tau={tau:.6g} (q={cfg['tau_q']}) "
          f"majors={int(major.sum())}/{P['n']} rank={cfg['rank']}", flush=True)
    C = ca.build_compressed(P, major, cfg["rank"])
    rep_lbl = f"Existing (tau={tau:.4g}, r={cfg['rank']})"
    xc_, tc_, itc_ = op_alm_compressed(P, C, a.tol)
    record(rep_lbl, "ALM", xc_, tc_, itc_, cert(xc_))

    print("[table2] compressed GP-signed (dense constraint penalised)", flush=True)
    xs_, ts_, its_, gs_ = GPC.solve_gp_signed(P, C, cert, tol=a.tol, max_seconds=600)
    record(rep_lbl, "GP-signed (penalised)", xs_, ts_, its_, gs_)

    out = Path(a.out) if a.out else HERE.parent / "results" / f"table2_consistency_{a.net}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}", flush=True)

    # ---- Gate 1 / Gate 2
    self_gap = rows[0]["Gap_F_pct"]
    g1 = abs(self_gap) < 1e-9 and abs(rows[0]["link_R2"] - 1.0) < 1e-12
    neg = [r for r in rows if r["Gap_F_pct"] < -1e-6]
    print(f"GATE 1 (reference self-evaluates to zero): {'PASS' if g1 else 'FAIL'} "
          f"(Gap_F={self_gap:.2e}%, R2={rows[0]['link_R2']:.12f})")
    print(f"GATE 2 (no negative Gap_F): {'PASS' if not neg else 'FAIL'}"
          + ("" if not neg else f" -- {[(r['method'], r['Gap_F_pct']) for r in neg]}"))
    return 0 if (g1 and not neg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
