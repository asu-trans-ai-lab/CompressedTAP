"""Experiment 10b -- TRAJECTORY test: does locality remove the floor?

exp10 compared representation classes by ONE step from the k=4 plateau. That establishes
gradient capture and the one-step frontier, but floor REMOVAL is a statement about where a
complete trajectory settles. This driver runs full compressed ALM trajectories from the
SAME iteration 0 as exp08 (w0 = x_start[minor], z = 0, lam = 0, rho = 100) for:

  global_r6      the incumbent (baseline; exp08 plateau ~0.0134)
  origin_light   per-origin weighted-SVD blocks, alpha=0.05, rmax=4   (R ~ 1,081)
  origin_med     per-origin weighted-SVD blocks, alpha=0.15, rmax=12  (R ~ 2,792)
  odx1g          ONE exchange atom per eligible OD, selected by INITIAL PATH COST:
                 a_q = (e_cheapest - e_costliest)/sqrt(2) over the OD's minors --
                 the steepest-descent route-substitution direction, replacing exp10's
                 flow-ranked odx1 which captured little (placement > count)

Matched settings across arms: TOL, rho schedule, mu hinge (fresh), inner maxiter 200 --
EXCEPT arms with R > 10k get --inner-big (default 400), the cap-sensitivity check exp10
flagged (all one-step runs exhausted 200 inner iterations, penalising large R).

Full-space reference from exp08: G = 0.00069 at k=10.

    python drivers/exp10b_trajectory.py [--case K10] [--scenario d05] [--outers 8]
"""
import argparse
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
from exp08_lockstep import CASES, CommonEvaluator, CompStepper
from exp10_structured import build_origin_blocks, origins_of_kept_ods


def build_odx_cost(od_of_minor, c_minor):
    """One exchange atom per eligible OD: cheapest minus costliest minor by initial cost."""
    order = np.argsort(od_of_minor, kind="stable")
    rows_all, cols_all, data_all = [], [], []
    col0 = 0
    i, n = 0, len(od_of_minor)
    while i < n:
        j = i
        while j < n and od_of_minor[order[j]] == od_of_minor[order[i]]:
            j += 1
        rows = order[i:j]
        i = j
        if len(rows) < 2:
            continue
        cheap = rows[np.argmin(c_minor[rows])]
        costly = rows[np.argmax(c_minor[rows])]
        if cheap == costly:
            continue
        rows_all.extend([int(cheap), int(costly)])
        cols_all.extend([col0, col0])
        data_all.extend([1 / np.sqrt(2), -1 / np.sqrt(2)])
        col0 += 1
    return sp.csr_matrix((data_all, (rows_all, cols_all)), shape=(n, col0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="K10", choices=list(CASES))
    ap.add_argument("--scenario", default="d05")
    ap.add_argument("--outers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--inner-big", type=int, default=400)
    ap.add_argument("--big-R", type=int, default=10000)
    a = ap.parse_args()
    dsdir, pool, tau = CASES[a.case]
    tag = f"{a.case}_{a.scenario}"
    rows_out = []
    K.banner(f"exp10b TRAJECTORIES from iteration 0 ({tag}, {a.outers} outers/arm)\n"
             f"  full-space reference (exp08): G = 0.00069 at k=10")

    P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
    if a.scenario.startswith("d") and a.scenario != "base":
        delta = int(a.scenario[1:]) / 100.0
        rng = np.random.default_rng(100)
        P["d"] = P["d"] * (1.0 + rng.uniform(-delta, delta, P["d"].size))
    x_start = ca.project_feasible(np.maximum(P["x0"], 1e-3), P["A"], P["d"], P["p2od"])
    major = ca.split_major_minor(P, tau=tau)
    P2 = dict(P)
    P2["x0"] = x_start
    Cb0 = ca.build_compressed(P2, major, a.rank, weight_flows=x_start)
    ev = CommonEvaluator(P)

    minor_idx = np.where(Cb0["minor"])[0]
    B2 = P["B"][Cb0["minor"]]
    A2 = P["A"][:, Cb0["minor"]]
    x0m = x_start[minor_idx]
    d_eff = P["d"] - np.asarray(A2 @ x0m).flatten()
    v_base = P.get("v0", 0.0) + np.asarray(B2.T @ x0m).flatten()
    od_of_minor = P["p2od"][minor_idx]
    od_o = origins_of_kept_ods(P)
    origin_of_minor = od_o[od_of_minor]
    wgt = np.sqrt(np.maximum(x0m, 0.0)) + 1e-9
    # initial path costs for the cost-selected exchange atoms
    v0c = P.get("v0", 0.0) + np.asarray(P["B"].T @ x_start).flatten()
    c_init = np.asarray(P["B"] @ P["bpr"].t(v0c)).flatten()

    def make_arm(name):
        if name == "global_r6":
            return sp.csr_matrix(Cb0["U"])
        if name == "origin_light":
            return build_origin_blocks(B2, wgt, origin_of_minor, 0.05, 4, P["m"])
        if name == "origin_med":
            return build_origin_blocks(B2, wgt, origin_of_minor, 0.15, 12, P["m"])
        if name == "odx1g":
            return build_odx_cost(od_of_minor, c_init[minor_idx])
        raise ValueError(name)

    for name in ("global_r6", "origin_light", "origin_med", "odx1g"):
        t0 = time.perf_counter()
        U = make_arm(name)
        t_build = time.perf_counter() - t0
        R = U.shape[1]
        inner = a.inner_big if R > a.big_R else a.inner
        D = (B2.T @ U)
        Mm = (A2 @ U)
        Cb = dict(B1=Cb0["B1"], A1=Cb0["A1"], U=U, D=D, M=Mm, d_eff=d_eff,
                  v_base=v_base, x0m=x0m, major=Cb0["major"], minor=Cb0["minor"],
                  r=R, svd_time=t_build)
        st = CompStepper(P, Cb, x_start, C.TOL, inner)
        Gs, cpus = [], []
        for k in range(1, a.outers + 1):
            rep = st.step()
            m = ev(st.export_x())
            Gs.append(m["G"])
            cpus.append(rep["cpu_s"])
            print(f"  {name:14s} k={k}  G={m['G']:.5f}  od={m['r_od']:.5f}  "
                  f"cpu={rep['cpu_s']:6.1f}s  nit={rep['nit']:>4}", flush=True)
            rows_out.append(dict(arm=name, R=R, inner_cap=inner, k=k,
                                 G=round(m["G"], 6), r_od=round(m["r_od"], 6),
                                 cpu_s=round(rep["cpu_s"], 1), nit=rep["nit"],
                                 build_s=round(t_build, 1)))
            K.write(f"exp10b_trajectory_{tag}", rows_out)
            if st.done:
                break
        print(f"  {name:14s} PLATEAU ~ {min(Gs):.5f}   total cpu {sum(cpus):.0f}s "
              f"(build {t_build:.0f}s, R={R:,}, inner cap {inner})\n", flush=True)
        rows_out.append(dict(arm=name, R=R, inner_cap=inner, k="min",
                             G=round(min(Gs), 6), cpu_s=round(sum(cpus), 1),
                             build_s=round(t_build, 1)))
        K.write(f"exp10b_trajectory_{tag}", rows_out)

    print(f"wrote {C.RESULTS / f'exp10b_trajectory_{tag}.csv'}\n[exp10b] DONE", flush=True)


if __name__ == "__main__":
    main()
