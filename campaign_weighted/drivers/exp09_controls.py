"""Experiment 9 -- certification controls for the exp08 lockstep plateau.

exp08 established: under a matched ALM architecture and identical iteration 0, the rank-6
compressed trajectory plateaus at common gap G ~ 1.34e-2 while full-space ALM passes 7e-4.
It does NOT yet separate three candidate causes, because after the first outer iteration
lam_F != lam_C (different subproblems from k=2 on) and the hinge term exists only on the
compressed side. The controls, all branched from the SAME in-process k=4 state:

  C1  SHARED-DUAL REPLAY   both sides get the same (lam, rho) -- the full side's, and, as a
      second variant, the compressed side's -- and each solves ONE primal subproblem with no
      multiplier update. If compressed still cannot reduce G, dual drift is exonerated.
  C2  HINGE ACCOUNTING     H(x_C) and ||grad_z H|| at k=4 (plus the per-step hinge series),
      and one compressed step with the hinge disabled (mu=0, c2=1e-12) at fixed duals.
      If H, its gradient, and the no-hinge G are all unchanged, the hinge is exonerated.
  C3  GRADIENT GEOMETRY    at the plateaued compressed iterate: g_T = OD-tangent-projected
      path-cost gradient (c minus its per-OD mean); eta_captured = (||g_T[major]||^2 +
      ||U' g_T[minor]||^2) / ||g_T||^2, plus the minor-block-only ratio. eta_missing ~ 1
      is the clean geometric explanation.
  C4  FULL-SPACE ESCAPE    one full-space ALM step started AT the decoded compressed
      iterate with the COMPRESSED side's (lam, rho) -- the only change is the
      representation -- against a paired normal compressed step from the same state.
  C5  RANK REPLAY          compressed-only trajectories from the same iteration 0 with
      r = 10 and r = 20 (fresh bases, same weights): does the plateau move with rank?

Basis-reuse note: exp08's basis was built FRESH from x_start on the perturbed instance, so
a "reused-basis floor" is already excluded for that run; C5 tests rank dependence only.

Checkpoint caveat: exp08's ck_*.npz store z in the coordinates of that run's ARPACK basis
(random start vector => column signs are not reproducible across processes), so this driver
re-runs the 4 lockstep outers in-process and branches from its own state.

    python drivers/exp09_controls.py [--case K10] [--scenario d05] [--branch-k 4]
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K
import config as C

import compressed_assignment as ca
from exp08_lockstep import CASES, CommonEvaluator, FullStepper, CompStepper


def clone_full(f):
    g = FullStepper.__new__(FullStepper)
    g.P, g.tol, g.maxit = f.P, f.tol, f.maxit
    g.x, g.lam = f.x.copy(), f.lam.copy()
    g.rho, g.prev, g.done = f.rho, f.prev, False
    return g


def clone_comp(c):
    g = CompStepper.__new__(CompStepper)
    g.P, g.C, g.tol, g.maxit, g.c2 = c.P, c.C, c.tol, c.maxit, c.c2
    g.s_dim, g.r = c.s_dim, c.r
    g.y, g.z = c.y.copy(), c.z.copy()
    g.lam, g.mu = c.lam.copy(), c.mu.copy()
    g.rho, g.prev, g.done = c.rho, c.prev, False
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="K10", choices=list(CASES))
    ap.add_argument("--scenario", default="d05")
    ap.add_argument("--branch-k", type=int, default=4)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--replay-ranks", default="10,20")
    ap.add_argument("--replay-outers", type=int, default=6)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--controls", default="C1,C2,C3,C4,C5",
                    help="subset to run, e.g. C3,C4 (skips the full-space lockstep arm "
                         "in the reproduction when C1 is not requested)")
    a = ap.parse_args()
    ctr = {c.strip().upper() for c in a.controls.split(",") if c.strip()}
    need_full = "C1" in ctr        # only C1 needs the full side's own trajectory/duals
    dsdir, pool, tau = CASES[a.case]
    tag = f"{a.case}_{a.scenario}"
    rows = []
    K.banner(f"exp09 CONTROLS {tag}: branch at k={a.branch_k}, base rank r={a.rank}")

    # ---- rebuild the exp08 configuration in-process
    P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
    if a.scenario.startswith("d") and a.scenario != "base":
        delta = int(a.scenario[1:]) / 100.0
        rng = np.random.default_rng(100)
        P["d"] = P["d"] * (1.0 + rng.uniform(-delta, delta, P["d"].size))
    x_start = ca.project_feasible(np.maximum(P["x0"], 1e-3), P["A"], P["d"], P["p2od"])
    major = ca.split_major_minor(P, tau=tau)
    P2 = dict(P)
    P2["x0"] = x_start
    Cb = ca.build_compressed(P2, major, a.rank, weight_flows=x_start)
    ev = CommonEvaluator(P)
    full = FullStepper(P, x_start, C.TOL, a.inner)
    comp = CompStepper(P, Cb, x_start, C.TOL, a.inner)
    assert float(np.max(np.abs(full.export_x() - comp.export_x()))) < 1e-9

    print(f"\n  reproduction of the lockstep to k={a.branch_k} "
          f"(ARPACK start vector differs across processes; expect close, not identical)")
    hinges = []
    for k in range(1, a.branch_k + 1):
        rf = full.step() if need_full else None
        rc = comp.step()
        mc = ev(comp.export_x())
        hinges.append(rc["hinge_term"])
        gtxt = ""
        row = dict(control="reproduction", step=k, G_comp=mc["G"],
                   hinge_comp=rc["hinge_term"])
        if rf is not None:
            mf = ev(full.export_x())
            gtxt = f"G_full={mf['G']:.5f}  "
            row["G_full"] = mf["G"]
        print(f"    k={k}  {gtxt}G_comp={mc['G']:.5f}  "
              f"hinge={rc['hinge_term']:.3e}", flush=True)
        rows.append(row)
    G_comp_base = mc["G"]
    K.write(f"exp09_controls_{tag}", rows)

    # ================================================================ C1
    if "C1" in ctr:
        print("\n  C1  SHARED-DUAL REPLAY (one primal subproblem, no multiplier update)")
        for src, lam_src, rho_src in (("lamF", full.lam, full.rho),
                                      ("lamC", comp.lam, comp.rho)):
            fc, cc = clone_full(full), clone_comp(comp)
            fc.lam, fc.rho = lam_src.copy(), rho_src
            cc.lam, cc.rho = lam_src.copy(), rho_src
            rf, rc = fc.step(), cc.step()      # post-step dual mutations are discarded
            mf, mc = ev(fc.export_x()), ev(cc.export_x())
            print(f"    common {src}: G_full={mf['G']:.5f}  G_comp={mc['G']:.5f}  "
                  f"(comp plateau before: {G_comp_base:.5f})", flush=True)
            rows.append(dict(control=f"C1_shared_dual_{src}", G_full=mf["G"],
                             G_comp=mc["G"], cpu_full=round(rf["cpu_s"], 1),
                             cpu_comp=round(rc["cpu_s"], 1)))
        K.write(f"exp09_controls_{tag}", rows)

    # ================================================================ C2
    if "C2" in ctr:
        print("\n  C2  HINGE ACCOUNTING")
        u4 = Cb["x0m"] + Cb["U"] @ comp.z
        phi4 = np.maximum(0.0, comp.mu - comp.c2 * u4)
        H4 = float(np.sum(phi4 ** 2 - comp.mu ** 2)) / (2 * comp.c2)
        gH4 = float(np.linalg.norm(Cb["U"].T @ phi4))
        print(f"    per-step hinge terms k=1..{a.branch_k}: "
              + ", ".join(f"{h:.3e}" for h in hinges))
        print(f"    at k={a.branch_k}: H={H4:.6e}  ||grad_z H||={gH4:.6e}  "
              f"min(u)={float(u4.min()):.3e}")
        cc = clone_comp(comp)
        cc.mu = np.zeros_like(cc.mu)
        cc.c2 = 1e-12                           # hinge numerically absent
        rc = cc.step()
        mc = ev(cc.export_x())
        print(f"    no-hinge step at fixed duals: G_comp={mc['G']:.5f} "
              f"(with hinge, paired control below)", flush=True)
        rows.append(dict(control="C2_hinge", H_at_k=H4, gradH_at_k=gH4,
                         G_comp_nohinge=mc["G"]))
        K.write(f"exp09_controls_{tag}", rows)

    # ================================================================ C4 (+ paired comp)
    x_c4 = comp.export_x()
    if "C4" in ctr:
        print("\n  C4  FULL-SPACE ESCAPE from the compressed iterate (comp's lam, rho)")
        esc = FullStepper(P, np.maximum(x_c4, 0.0), C.TOL, a.inner)
        esc.lam, esc.rho = comp.lam.copy(), comp.rho
        r_esc = esc.step()
        m_esc = ev(esc.export_x())
        cc = clone_comp(comp)
        r_cc = cc.step()
        m_cc = ev(cc.export_x())
        print(f"    escape (full step from x_C):   G={m_esc['G']:.5f}  "
              f"cpu={r_esc['cpu_s']:.1f}s")
        print(f"    paired normal compressed step: G={m_cc['G']:.5f}  "
              f"cpu={r_cc['cpu_s']:.1f}s", flush=True)
        rows.append(dict(control="C4_escape", G_escape=m_esc["G"],
                         G_comp_paired=m_cc["G"],
                         cpu_escape=round(r_esc["cpu_s"], 1),
                         cpu_comp=round(r_cc["cpu_s"], 1)))
        K.write(f"exp09_controls_{tag}", rows)

    # ================================================================ C3
    if "C3" not in ctr:
        x = None
    else:
        print("\n  C3  GRADIENT GEOMETRY at the plateaued compressed iterate")
        x = np.maximum(x_c4, 0.0)
        v = P.get("v0", 0.0) + np.asarray(P["B"].T @ x).flatten()
        c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
        cnt = np.bincount(P["p2od"], minlength=P["n_od"]).astype(float)
        odmean = (np.bincount(P["p2od"], weights=c, minlength=P["n_od"])
                  / np.maximum(cnt, 1))
        gT = c - odmean[P["p2od"]]              # OD-conservation tangent projection
        g_maj = gT[Cb["major"]]
        g_min = gT[Cb["minor"]]
        cap_min = float(np.linalg.norm(Cb["U"].T @ g_min) ** 2)
        n_min = float(np.linalg.norm(g_min) ** 2)
        n_tot = float(np.linalg.norm(gT) ** 2)
        eta_cap = (float(np.linalg.norm(g_maj) ** 2) + cap_min) / n_tot
        eta_cap_minor = cap_min / max(n_min, 1e-300)
        print(f"    ||g_T||^2 split: majors {np.linalg.norm(g_maj)**2/n_tot:5.1%}, "
              f"minors {n_min/n_tot:5.1%}")
        print(f"    eta_captured (total)      = {eta_cap:.4f}   "
              f"eta_missing = {1-eta_cap:.4f}")
        print(f"    eta_captured (minor-only) = {eta_cap_minor:.6f}  "
              f"-> span(U_{Cb['r']}) sees {eta_cap_minor:.2%} of the minor-block "
              f"equilibrium gradient", flush=True)
        rows.append(dict(control="C3_geometry", eta_captured=eta_cap,
                         eta_missing=1 - eta_cap, eta_captured_minor=eta_cap_minor,
                         minor_share_of_gT=n_min / n_tot))
        K.write(f"exp09_controls_{tag}", rows)

    # ================================================================ C5
    if "C5" in ctr:
        print("\n  C5  RANK REPLAY from iteration 0 (compressed-only trajectories)")
        for r in [int(x) for x in a.replay_ranks.split(",") if x.strip()]:
            Cr = ca.build_compressed(P2, major, r, weight_flows=x_start)
            cr = CompStepper(P, Cr, x_start, C.TOL, a.inner)
            Gs = []
            for k in range(1, a.replay_outers + 1):
                rc = cr.step()
                Gs.append(ev(cr.export_x())["G"])
            print(f"    r={r:<3d} G: " + " ".join(f"{g:.5f}" for g in Gs)
                  + f"   plateau~{min(Gs):.5f}", flush=True)
            rows.append(dict(control=f"C5_rank_r{r}", plateau_G=min(Gs),
                             G_series=";".join(f"{g:.5f}" for g in Gs)))
            K.write(f"exp09_controls_{tag}", rows)

    print(f"\nwrote {C.RESULTS / f'exp09_controls_{tag}.csv'}\n[exp09] DONE", flush=True)


if __name__ == "__main__":
    main()
