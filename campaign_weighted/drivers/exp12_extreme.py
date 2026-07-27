"""Experiment 12, stage 2 -- the path-rich, OD-light extreme case: break-even vs richness.

Fixed OD set (top-M by demand), nested pools X8 c X16 c X32 c X64 (exp12_gen_pools).
For each depth, the operator-matched pair under the lockstep instrument:

  FULL        full-space ALM (FullStepper), one outer at a time
  COMPRESSED  x_minor = w0 + U_G z + A alpha:
                global weighted-SVD backbone, r = 6 (small on purpose);
                at most B = 500 dynamically priced OD exchange atoms
                (cost-selected cheapest-vs-costliest over used minors, coefficient-
                bounded); REPRICED ONLY WHEN PROGRESS STALLS (relative G improvement
                below --stall-frac between outers), otherwise atoms are kept

Both sides scored per outer by the independent CommonEvaluator; the pre-registered
comparison is TIME-TO-TARGET at moderate accuracy G* in {0.020, 0.015} -- NOT final wall
time after both sides stall, and no claim below the certified representation floor.

  S(G*) = T_full(G*) / T_compressed(G*)          expected: rising with K

Also recorded per outer: cpu, inner iterations, OD residual, decoder/OD/link kernel CPU,
and (if psutil is present) peak RSS per arm.

    python drivers/exp12_extreme.py [--M 20000] [--depths 8,16,32,64] [--outers 12]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K
import config as C

import compressed_assignment as ca
from exp08_lockstep import CommonEvaluator, FullStepper, CompStepper
from exp11_hybrid import HybridStepper, od_gap_and_costs, build_atoms

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "extreme_case"
TARGETS = (0.020, 0.015)

try:
    import psutil
    RSS = lambda: psutil.Process().memory_info().rss / 2**20
except Exception:
    RSS = lambda: float("nan")


def run_depth(depth, ds, a, rows):
    pool = f"pool_X{depth}.csv"
    P = ca.load_problem(str(ds), pool, multi_path_only=True)
    ev = CommonEvaluator(P)
    x_start = ca.project_feasible(np.maximum(P["x0"], 1e-3), P["A"], P["d"], P["p2od"])
    kbar = P["n"] / P["n_od"]
    K.banner(f"exp12 X{depth}: n={P['n']:,} od={P['n_od']:,} Kbar={kbar:.2f}")

    def log(arm, k, m, rep, cpu_tot, extra=None):
        row = dict(depth=depth, arm=arm, k=k, G=round(m["G"], 6),
                   r_od=round(m["r_od"], 6), cpu_s=round(rep["cpu_s"], 1),
                   cpu_cum_s=round(cpu_tot, 1), nit=rep["nit"],
                   rss_mb=round(RSS(), 0))
        for kn, tv in rep.get("kt", {}).items():
            row[f"t_{kn}"] = round(tv, 2)
        if extra:
            row.update(extra)
        rows.append(row)
        K.write(f"exp12_extreme_{a.tag}", rows)

    hit = {arm: {t: None for t in TARGETS} for arm in ("full", "comp")}

    # ---------------------------------------------------------------- FULL arm
    full = FullStepper(P, x_start, C.TOL, a.inner)
    cpu_tot = 0.0
    for k in range(1, a.outers + 1):
        rep = full.step()
        cpu_tot += rep["cpu_s"]
        m = ev(full.export_x())
        for t in TARGETS:
            if hit["full"][t] is None and m["G"] <= t:
                hit["full"][t] = round(cpu_tot, 1)
        print(f"  full  k={k:2d}  G={m['G']:.5f}  od={m['r_od']:.5f}  "
              f"cpu={rep['cpu_s']:7.1f}s  nit={rep['nit']:>4}", flush=True)
        log("full", k, m, rep, cpu_tot)
        if full.done or (min(hit["full"][t] is not None for t in TARGETS)
                         and m["G"] < 0.012):
            break

    # ---------------------------------------------------------- COMPRESSED arm
    minor = None
    major = ca.split_major_minor(P, tau=np.quantile(x_start, 0.75))
    # tau by quantile of the start flows keeps the split meaningful across depths;
    # every OD keeps its max-flow path explicit (split_major_minor guarantees it)
    P2 = dict(P)
    P2["x0"] = x_start
    Cb0 = ca.build_compressed(P2, major, a.rank, weight_flows=x_start)
    minor_idx = np.where(Cb0["minor"])[0]
    B2 = P["B"][Cb0["minor"]]
    A2 = P["A"][:, Cb0["minor"]]
    od_of_minor = P["p2od"][minor_idx]
    red0 = 100.0 * (P["n"] - int(major.sum()) - Cb0["r"]) / P["n"]
    print(f"  comp: majors={int(major.sum()):,} r={Cb0['r']} "
          f"reduction={red0:.1f}% (before atoms)", flush=True)

    x_now = x_start.copy()
    lam = np.zeros(P["n_od"])
    rho, prev = 100.0, np.inf
    cpu_tot, G_prev = 0.0, np.inf
    atoms_A, bnd = sp.csr_matrix((len(minor_idx), 0)), []
    n_reprices = 0
    for k in range(1, a.outers + 1):
        x0m = np.maximum(x_now, 0.0)[minor_idx]
        d_eff = P["d"] - np.asarray(A2 @ x0m).flatten()
        v_base = P.get("v0", 0.0) + np.asarray(B2.T @ x0m).flatten()
        stalled = (G_prev - ev(x_now)["G"]) < a.stall_frac * G_prev if k > 1 else True
        if stalled:
            Gq, c = od_gap_and_costs(P, x_now)
            active = np.argsort(Gq)[::-1][:a.B]
            atoms_A, bnd = build_atoms(od_of_minor, x0m, c[minor_idx], active, 1)
            n_reprices += 1
        U = sp.hstack([sp.csr_matrix(Cb0["U"]), atoms_A]).tocsr()
        Cb = dict(B1=Cb0["B1"], A1=Cb0["A1"], U=U, D=(B2.T @ U), M=(A2 @ U),
                  d_eff=d_eff, v_base=v_base, x0m=x0m, major=Cb0["major"],
                  minor=Cb0["minor"], r=U.shape[1], svd_time=0.0)
        st = HybridStepper(P, Cb, np.maximum(x_now, 0.0), C.TOL, a.inner,
                           [(None, None)] * Cb0["r"] + bnd)
        st.lam, st.rho, st.prev = lam.copy(), rho, prev
        rep = st.step()
        lam, rho, prev = st.lam.copy(), st.rho, st.prev
        x_now = st.export_x()
        cpu_tot += rep["cpu_s"]
        m = ev(x_now)
        for t in TARGETS:
            if hit["comp"][t] is None and m["G"] <= t:
                hit["comp"][t] = round(cpu_tot, 1)
        G_prev = m["G"]
        print(f"  comp  k={k:2d}  G={m['G']:.5f}  od={m['r_od']:.5f}  "
              f"cpu={rep['cpu_s']:7.1f}s  nit={rep['nit']:>4}  "
              f"atoms={atoms_A.shape[1]:,}{' *repriced' if stalled else ''}", flush=True)
        log("comp", k, m, rep, cpu_tot,
            dict(atoms=atoms_A.shape[1], repriced=int(stalled)))
        if all(hit["comp"][t] is not None for t in TARGETS):
            break

    # ---------------------------------------------------------------- verdict
    line = dict(depth=depth, arm="SUMMARY", kbar=round(kbar, 2), n=P["n"],
                n_od=P["n_od"], reduction_pct=round(red0, 1), reprices=n_reprices)
    for t in TARGETS:
        tf, tc = hit["full"][t], hit["comp"][t]
        line[f"T_full@{t}"] = tf
        line[f"T_comp@{t}"] = tc
        line[f"S@{t}"] = round(tf / tc, 2) if tf and tc else None
        print(f"  X{depth}  G*={t}:  T_full={tf}  T_comp={tc}  "
              f"S={line[f'S@{t}']}", flush=True)
    rows.append(line)
    K.write(f"exp12_extreme_{a.tag}", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="sketch", choices=["sketch", "regional"])
    ap.add_argument("--M", type=int, default=20000)
    ap.add_argument("--depths", default="8,16,32,64")
    ap.add_argument("--outers", type=int, default=12)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--stall-frac", type=float, default=0.10)
    a = ap.parse_args()
    ds = SCRATCH / a.net / f"M{a.M}"
    if a.net == "sketch" and not (ds / "pool_X8.csv").exists():
        ds = SCRATCH / f"M{a.M}"          # pre---net layout
    a.tag = f"{a.net}_M{a.M}"
    assert (ds / "pool_X8.csv").exists(), "run exp12_gen_pools.py first"
    rows = []
    for depth in [int(x) for x in a.depths.split(",")]:
        run_depth(depth, ds, a, rows)
    print(f"\nwrote {C.RESULTS / f'exp12_extreme_{a.tag}.csv'}\n[exp12] DONE", flush=True)


if __name__ == "__main__":
    main()
