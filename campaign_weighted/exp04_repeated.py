"""Experiment 4 -- the repeated-scenario use case, and the break-even count.

This is the experiment the reviewers asked for. Referee 1 objected that the nominal flow is
obtained by solving the full problem, so compression is circular; Referee 2 objected that a
path-based method could simply warm-start from the previous solution and get the same benefit.
Both objections are about the SAME setting: a baseline scenario followed by several nearby
ones. So we run exactly that setting and measure it.

Protocol, all on the grid family so the perturbation is controlled:

  BASELINE (paid once)
    solve the baseline scenario uncompressed        -> nominal flow w0, and the warm start
    build the compressed basis from it              -> Phi_r, D, M       [preprocessing]

  EACH FOLLOW-UP SCENARIO (demand or capacity perturbed by +/- delta)
    (a) uncompressed ALM, WARM-STARTED from the baseline solution   <- the reviewer's proposal
    (b) compressed ALM reusing the baseline basis, unchanged        <- ours
    both measured to the same tolerance, and both scored against a cold uncompressed
    reference solved on that scenario, so accuracy is comparable.

Reported per scenario: time and feasible objective gap and link-flow error; then aggregated
into the break-even count

    H* = ceil( T_preprocessing / (T_warm_uncompressed - T_compressed) )

which is the number of follow-up solves needed to repay building the basis. If the denominator
is non-positive the compressed solve is not faster and no H* exists; that is reported as such
rather than hidden.

    python drivers/exp04_repeated.py [--N 24] [--K 32] [--deltas 0.05,0.10,0.20]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C
from exp03_grid import gen_grid, logit_x0, SCRATCH

import compressed_assignment as ca
import v3_metrics as M


def perturb(gdir, out_dir, kind, delta, seed):
    """Copy the grid and perturb demand (volumes) or capacity (link capacities)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for f in ("node.csv", "link.csv", "demand.csv"):
        pd.read_csv(gdir / f).to_csv(out_dir / f, index=False)
    if kind == "demand":
        d = pd.read_csv(out_dir / "demand.csv")
        d["volume"] = d["volume"] * (1.0 + rng.uniform(-delta, delta, len(d)))
        d.to_csv(out_dir / "demand.csv", index=False)
    else:
        l = pd.read_csv(out_dir / "link.csv")
        l["capacity"] = l["capacity"] * (1.0 + rng.uniform(-delta, delta, len(l)))
        l.to_csv(out_dir / "link.csv", index=False)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--deltas", default="0.05,0.10,0.20")
    ap.add_argument("--rank", type=int, default=C.GRID_RANK)
    a = ap.parse_args()
    deltas = [float(x) for x in a.deltas.split(",")]
    rows = []

    # ---------------------------------------------------------------- baseline
    base = SCRATCH / f"rep_g{a.N}"
    m, n_od, q = gen_grid(a.N, base)
    pool = base / f"pool_K{a.K}.csv"
    r = subprocess.run([str(C.KSP), str(base), str(a.K), "1.4", str(pool)],
                       capture_output=True, text=True, env=C.env())
    assert r.returncode == 0, r.stderr[-300:]
    df = logit_x0(pool, q)
    tau = float(np.quantile(df["volume_ref"].to_numpy(float), C.GRID_TAU_Q))
    K.banner(f"exp04 baseline: grid N={a.N} K={a.K}  n={len(df):,}  tau={tau:.4f}")

    P0 = ca.load_problem(str(base), pool.name, multi_path_only=True)
    t = time.perf_counter()
    s0 = ca.solve_full(P0, tol=C.TOL, max_outer=C.MAX_OUTER_FULL)
    t_nominal = time.perf_counter() - t          # PREPROCESSING part 1: the nominal flow
    x_warm = s0["x_raw"].copy()
    P0["x0"] = np.maximum(x_warm, 0.0)           # nominal flow = the baseline solution
    major = ca.split_major_minor(P0, tau=tau)
    t = time.perf_counter()
    Cb = ca.build_compressed(P0, major, a.rank, weight_flows=P0["x0"])
    t_svd = time.perf_counter() - t              # PREPROCESSING part 2: the basis
    s_maj = int(major.sum())
    red = 100.0 * (P0["n"] - s_maj - Cb["r"]) / P0["n"]
    print(f"  nominal flow {t_nominal:7.2f}s | basis {t_svd:6.2f}s | "
          f"majors {s_maj:,} r={Cb['r']} red={red:.1f}%", flush=True)
    rows.append(dict(scenario="baseline", kind="-", delta=0.0,
                     t_nominal_s=round(t_nominal, 2), t_svd_s=round(t_svd, 2)))

    # ---------------------------------------------------------------- follow-ups
    for kind in ("demand", "capacity"):
        for i, dl in enumerate(deltas):
            sdir = perturb(base, SCRATCH / f"rep_g{a.N}_{kind}{int(dl*100)}", kind, dl, 100 + i)
            (sdir / pool.name).write_bytes((base / pool.name).read_bytes())
            Ps = ca.load_problem(str(sdir), pool.name, multi_path_only=True)

            # cold uncompressed reference on this scenario (accuracy yardstick)
            t = time.perf_counter()
            sc_ref = ca.solve_full(Ps, tol=C.TOL, max_outer=C.MAX_OUTER_FULL)
            t_cold = time.perf_counter() - t
            xr = M.convert_euclid(sc_ref["x_raw"], Ps["d"], Ps["p2od"])
            v_ref = M.link_flow(Ps, xr); o_ref = float(Ps["bpr"].beckmann(v_ref))

            # (a) uncompressed, WARM-STARTED from the baseline solution
            Pw = dict(Ps); Pw["x0"] = np.maximum(x_warm, 0.0)
            t = time.perf_counter()
            sw = ca.solve_full(Pw, tol=C.TOL, max_outer=C.MAX_OUTER_FULL)
            t_warm = time.perf_counter() - t
            xw = M.convert_euclid(sw["x_raw"], Ps["d"], Ps["p2od"])
            vw = M.link_flow(Ps, xw); ow = float(Ps["bpr"].beckmann(vw))

            # (b) compressed, REUSING the baseline basis unchanged
            Cs = dict(Cb)
            Cs["d_eff"] = Ps["d"] - np.asarray(Ps["A"][:, Cb["minor"]] @ Cb["x0m"]).flatten()
            t = time.perf_counter()
            scp = ca.solve_compressed(Ps, Cs, regime="hard", tol=C.TOL,
                                      max_outer=C.MAX_OUTER_COMP)
            t_comp = time.perf_counter() - t
            xc = M.convert_euclid(scp["x_raw"], Ps["d"], Ps["p2od"])
            vc = M.link_flow(Ps, xc); oc = float(Ps["bpr"].beckmann(vc))

            best = min(o_ref, ow, oc)
            lw = 100 * np.linalg.norm(vw - v_ref) / max(np.linalg.norm(v_ref), 1e-12)
            lc = 100 * np.linalg.norm(vc - v_ref) / max(np.linalg.norm(v_ref), 1e-12)
            print(f"  {kind:8s} d={dl:4.2f} | cold {t_cold:7.2f}s | "
                  f"warm-full {t_warm:7.2f}s gap {100*(ow-best)/best:7.4f}% link {lw:6.3f}% | "
                  f"compressed {t_comp:7.2f}s gap {100*(oc-best)/best:7.4f}% link {lc:6.3f}%",
                  flush=True)
            rows.append(dict(scenario=f"{kind}{int(dl*100)}", kind=kind, delta=dl,
                             t_cold_s=round(t_cold, 2), t_warmfull_s=round(t_warm, 2),
                             t_compressed_s=round(t_comp, 2),
                             gap_warmfull_pct=round(100 * (ow - best) / best, 4),
                             gap_compressed_pct=round(100 * (oc - best) / best, 4),
                             linkerr_warmfull_pct=round(lw, 4),
                             linkerr_compressed_pct=round(lc, 4)))
            K.write("exp04_repeated", rows)

    # ---------------------------------------------------------------- break-even
    fu = [x for x in rows if x["scenario"] != "baseline"]
    tw = float(np.mean([x["t_warmfull_s"] for x in fu]))
    tc = float(np.mean([x["t_compressed_s"] for x in fu]))
    tp = t_nominal + t_svd
    print(f"\n  preprocessing  T_prep = {tp:.2f}s  (nominal {t_nominal:.2f} + basis {t_svd:.2f})")
    print(f"  per follow-up  warm-started uncompressed {tw:.2f}s   compressed {tc:.2f}s")
    if tw - tc > 0:
        H = int(np.ceil(tp / (tw - tc)))
        print(f"  saving per solve {tw - tc:.2f}s  ->  BREAK-EVEN H* = {H} follow-up solves")
    else:
        H = None
        print(f"  compressed is NOT faster per solve ({tw - tc:+.2f}s): no break-even exists")
    rows.append(dict(scenario="SUMMARY", t_prep_s=round(tp, 2),
                     t_warmfull_mean_s=round(tw, 2), t_compressed_mean_s=round(tc, 2),
                     saving_per_solve_s=round(tw - tc, 2), break_even_H=H if H else "none"))
    K.write("exp04_repeated", rows)
    print(f"\nwrote {C.RESULTS/'exp04_repeated.csv'}\n[exp04] DONE", flush=True)


if __name__ == "__main__":
    main()
