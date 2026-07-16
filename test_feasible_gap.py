#!/usr/bin/env python3
"""Regression test for the negative-BPR-gap defect (OPRE review 2026-07-08, R1 pt 2 / R2 pt 3).

    python test_feasible_gap.py

The defect: the reported 'BPR gap' was `bpr_objective(v_iterate) - bpr_objective(v_ref)` with
v_iterate taken at the RAW ALM iterate, which can violate OD conservation. An iterate that
under-delivers demand prices BELOW the optimum, so the gap goes negative -- and the old summary
even printed "(Better!)". The fix (feasible_projection) clips negative flows and rescales each
OD to its demand before evaluating, so the reported gap is taken at a FEASIBLE point, which by
convexity cannot price below the optimum.

Test instance (closed form, no solver needed for the reference):
    OD0: two identical parallel links (t0=10, cap=100), demand 100 -> UE split 50/50
    OD1: two identical parallel links (t0=5,  cap=200), demand 200 -> UE split 100/100
    OD2: singleton, its own link (t0=8, cap=150),       demand 60  -> carries 60
v_ref is the exact UE, so bpr_optimal is the true optimum and every feasible point must price
at or above it.

Checks:
    T1  ALM.__init__ builds the singleton background: v0[link4] == 60
    T2  feasible_projection restores feasibility exactly from a deliberately corrupted point
        (per-OD sums == demand to 1e-9, all flows >= 0, background included in v_f)
    T3  THE DEFECT, reproduced then fixed: at an iterate that under-delivers demand the RAW
        gap is negative; the FEASIBLE gap at the same iterate is >= 0
    T4  after optimize(), the reported bpr_gap_feas_pct is >= 0 against the exact reference
"""
import numpy as np
from scipy.sparse import csr_matrix

from compressed_tap import ALM, bpr_objective, decompose_paths, compute_svd_compression


def build_instance():
    # paths: p0(OD0,link0) p1(OD0,link1) p2(OD1,link2) p3(OD1,link3) p4(OD2,link4)
    B = csr_matrix(np.array([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]], dtype=float))
    capacity = np.array([100.0, 100.0, 200.0, 200.0, 150.0])
    t_0 = np.array([10.0, 10.0, 5.0, 5.0, 8.0])
    d = np.array([100.0, 200.0, 60.0])
    path_to_od = np.array([0, 0, 1, 1, 2])
    # x_ref only drives the major/minor split (60>50 major, 40<50 minor, 120/80 major)
    x_ref = np.array([60.0, 40.0, 120.0, 80.0, 60.0])
    od_info = {
        "od_pairs": [(1, 2), (3, 4), (5, 6)],
        "od_pair_to_idx": {(1, 2): 0, (3, 4): 1, (5, 6): 2},
        "path_to_od": path_to_od,
        "od_demand": d,
        "paths_by_od": {0: [0, 1], 1: [2, 3], 2: [4]},
        "n_od": 3,
    }
    # exact UE: equal split on identical parallel links; singleton carries its demand
    v_ue = np.array([50.0, 50.0, 100.0, 100.0, 60.0])
    return B, x_ref, v_ue, capacity, t_0, od_info


def main():
    B, x_ref, v_ue, capacity, t_0, od_info = build_instance()
    decomp = decompose_paths(B, x_ref, od_info, threshold=50.0)
    svd_dict = compute_svd_compression(decomp["B2"], decomp["w_ref"], max_rank=1,
                                       use_truncated_svd=True)
    alm = ALM(decomp, svd_dict, capacity, t_0, od_info, v_ue,
              c1_init=1e3, c2_init=1e3, beta_penalty=4.0, tolerance=1e-4)
    fails = []

    # T1 -- singleton background
    ok = alm.v0 is not None and abs(float(np.asarray(alm.v0).flatten()[4]) - 60.0) < 1e-12
    print("T1 singleton background v0[link4]==60 : %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        fails.append("T1")

    # T2 -- projection restores feasibility exactly from a corrupted point
    y_bad = decomp["y_ref"] * 1.07            # 7% over-delivery on majors
    w_bad = np.array([-5.0])                  # negative minor flow
    y_f, w_f, v_f = alm.feasible_projection(y_bad, w_bad)
    s0 = y_f[alm.od_of_major == 0].sum() + (w_f[alm.od_of_minor == 0].sum() if len(w_f) else 0)
    s1 = y_f[alm.od_of_major == 1].sum() + (w_f[alm.od_of_minor == 1].sum() if len(w_f) else 0)
    ok = (abs(s0 - 100.0) < 1e-9 and abs(s1 - 200.0) < 1e-9
          and y_f.min() >= 0 and (len(w_f) == 0 or w_f.min() >= 0)
          and abs(v_f[4] - 60.0) < 1e-12)
    print("T2 projection: OD sums exact, flows>=0, background in v_f : %s"
          % ("PASS" if ok else "FAIL"))
    if not ok:
        fails.append("T2")

    # T3 -- the defect, reproduced then fixed. Under-deliver demand by 40%.
    y_under = alm.A1.T @ (0.6 * alm.d_multi / np.asarray(alm.A1.sum(axis=1)).flatten())
    y_under = np.asarray(y_under).flatten()
    alm.w = np.zeros(decomp["n_minor"])
    v_under = (alm.v0 if alm.v0 is not None else 0.0) + alm.B1.T @ y_under
    v_under = np.asarray(v_under).flatten()
    raw_gap = bpr_objective(v_under, capacity, t_0) - alm.bpr_ref
    _, _, v_fp = alm.feasible_projection(y_under, alm.w)
    feas_gap = bpr_objective(v_fp, capacity, t_0) - alm.bpr_ref
    ok = raw_gap < 0 and feas_gap >= -1e-9
    print("T3 under-delivering iterate: raw gap %.2f < 0 (the defect), "
          "feasible gap %.6f >= 0 (the fix) : %s"
          % (raw_gap, feas_gap, "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("T3")

    # T4 -- end to end: reported feasible gap nonnegative against the EXACT reference
    result = alm.optimize(max_outer_iter=10, max_inner_iter=100, verbose=False)
    m = result["final_metrics"]
    ok = m["ref_obj_rel_diff"] >= -1e-9
    print("T4 optimize(): ref_obj_rel_diff (feasible) = %+.6f%% >= 0 vs exact UE : %s"
          % (m["ref_obj_rel_diff"], "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("T4")

    print("\n%s" % ("ALL CHECKS PASSED" if not fails else "FAILED: %s" % ", ".join(fails)))
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(main())
