"""Experiment 11 -- HYBRID global-local representation: origin backbone + active-OD
cost-selected exchange atoms, with coefficient bounds. The exp10b verdicts drive this:

  * origin blocks: class-level plateau ~0.0113 regardless of budget (backbone only);
  * odx1g: converges CLEANLY to its span optimum (no boundary churn, no cap-exhaustion)
    at 0.0117 -- exchange atoms have only a CAPTURE defect, so the fix is more atoms on
    the ODs that matter, repriced as costs move.

Representation:   x_minor = w0 + U_O z + A_A alpha
  U_O      origin-light backbone (alpha=0.05, rmax=4; R ~ 1,081), z unbounded;
  A_A      exchange atoms a_q = (e_cheap - e_costly)/sqrt(2) over USED minors of OD q
           (used: x >= eps), selected by per-OD disequilibrium
           G_q = sum_p x_p (c_p - c_q^min), top-B ODs only;
  alpha    BOX-BOUNDED per atom from available anchor flow:
             -sqrt(2) x0m[cheap] <= alpha_q <= sqrt(2) x0m[costly]
           so the atom part alone cannot drive either endpoint negative -- route
           substitution, not signed reconstruction. The hinge stays on the COMBINED
           decode (the backbone can still cross zero), but atoms need no hinge repair.

Arms (all 8 outers from the certified iteration 0; origin-light baseline = exp10b record):
  hyb500x1, hyb2000x1, hyb5000x1    static atoms, 1 per active OD, B = 500/2000/5000
  hyb2000x2                          static, 2 atoms per active OD (costliest->cheapest,
                                     2nd-costliest->cheapest)
  dyn2000x1                          DYNAMIC exchange pricing: every outer iteration,
                                     re-anchor w0 at the current decoded point, recompute
                                     G_q and atom pairs at current costs, reset (z, alpha,
                                     mu); lam and rho carry across (restricted-master
                                     scheme; the resets are disclosed, not hidden)

Reported per step: common G (independent evaluator), r_OD, cpu, nit, boundary fraction;
per arm: plateau and time-to-target for G in {1e-2, 7e-3, 5e-3}.

    python drivers/exp11_hybrid.py [--case K10] [--scenario d05] [--outers 8]
                                   [--arms hyb500x1,hyb2000x1,hyb5000x1,hyb2000x2,dyn2000x1]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K
import config as C

import compressed_assignment as ca
from exp08_lockstep import CASES, CommonEvaluator, CompStepper
from exp10_structured import build_origin_blocks, origins_of_kept_ods

EPS_USED = 1e-6


class HybridStepper(CompStepper):
    """CompStepper whose latent block carries per-coordinate bounds (atoms are boxed,
    backbone coordinates stay free). Only the bounds construction differs from the
    certified step(); the objective/gradient code is inherited unchanged via super().

    Implementation: scipy's L-BFGS-B receives bounds = major bounds + self.z_bounds."""

    def __init__(self, P, Cb, x_start, tol, maxit, z_bounds):
        super().__init__(P, Cb, x_start, tol, maxit)
        self.z_bounds = z_bounds

    def step(self):
        # identical to CompStepper.step except for the bounds list; kept in sync by
        # delegating everything possible to the parent's own fg machinery would require
        # refactoring the certified file, so the loop body is reproduced with the one
        # change (bounds) -- see CompStepper.step for provenance.
        P, Cb, c2 = self.P, self.C, self.c2
        bpr = P["bpr"]
        B1, A1, U, D, Mm = Cb["B1"], Cb["A1"], Cb["U"], Cb["D"], Cb["M"]
        d_eff, v_base, x0m = Cb["d_eff"], Cb["v_base"], Cb["x0m"]
        s_dim, r = self.s_dim, self.r
        lam, rho, mu = self.lam, self.rho, self.mu
        kt = dict(link=0.0, bpr=0.0, grad_major=0.0, grad_latent=0.0,
                  decode_hinge=0.0, od=0.0)

        def fg(yz):
            yv, zv = yz[:s_dim], yz[s_dim:]
            t1 = time.perf_counter()
            v = v_base + np.asarray(B1.T @ yv).flatten() + D @ zv
            t2 = time.perf_counter()
            rr = (np.asarray(A1 @ yv).flatten() + Mm @ zv) - d_eff
            t3 = time.perf_counter()
            f = bpr.beckmann(v) + lam @ rr + 0.5 * rho * float(rr @ rr)
            gv = bpr.t(v)
            t4 = time.perf_counter()
            gy = (np.asarray(B1 @ gv).flatten()
                  + np.asarray(A1.T @ (lam + rho * rr)).flatten())
            t5 = time.perf_counter()
            gz = D.T @ gv + Mm.T @ (lam + rho * rr)
            t6 = time.perf_counter()
            u = x0m + U @ zv
            phi = np.maximum(0.0, mu - c2 * u)
            f += float(np.sum(phi ** 2 - mu ** 2)) / (2 * c2)
            gz += -(U.T @ phi)
            t7 = time.perf_counter()
            kt["link"] += t2 - t1
            kt["od"] += t3 - t2
            kt["bpr"] += t4 - t3
            kt["grad_major"] += t5 - t4
            kt["grad_latent"] += t6 - t5
            kt["decode_hinge"] += t7 - t6
            return f, np.concatenate([gy, np.asarray(gz).flatten()])

        bounds = [(0, None)] * s_dim + self.z_bounds
        t0 = time.perf_counter()
        res = minimize(lambda w: fg(w)[0], np.concatenate([self.y, self.z]),
                       jac=lambda w: fg(w)[1], method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.maxit, "gtol": self.tol * 0.1})
        cpu = time.perf_counter() - t0
        self.y, self.z = res.x[:s_dim], res.x[s_dim:]
        rr = (np.asarray(A1 @ self.y).flatten() + Mm @ self.z) - d_eff
        cons = float(np.linalg.norm(rr))
        u_full = x0m + np.asarray(U @ self.z).flatten()
        Rplus = float(np.linalg.norm(np.minimum(u_full, 0.0)))
        rep = dict(cpu_s=cpu, nit=int(res.nit), aug=float(res.fun), cons=cons,
                   Rplus=Rplus, rho=rho, lam_norm=float(np.linalg.norm(lam)), kt=kt)
        self.mu = np.maximum(0.0, mu - c2 * u_full)
        self.lam = lam + rho * rr
        if cons < self.tol and Rplus < 1e-3:
            self.done = True
        else:
            if cons > 0.25 * self.prev:
                self.rho = min(rho * 2, 1e6)
            self.prev = cons
        return rep


def od_gap_and_costs(P, x, ev_cache=None):
    """Per-OD disequilibrium G_q and current path costs c at point x (clipped)."""
    xc = np.maximum(x, 0.0)
    v = P.get("v0", 0.0) + np.asarray(P["B"].T @ xc).flatten()
    c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
    ordr = np.lexsort((c, P["p2od"]))
    grp = np.searchsorted(P["p2od"][ordr], np.arange(P["n_od"]))
    cmin = c[ordr[grp]]
    Gq = np.bincount(P["p2od"], weights=xc * (c - cmin[P["p2od"]]),
                     minlength=P["n_od"])
    return Gq, c


def build_atoms(od_of_minor, x0m, c_minor, active_ods, per_od):
    """Cost-selected exchange atoms over USED minors for the active ODs.
    Returns (sparse A, bounds list)."""
    order = np.argsort(od_of_minor, kind="stable")
    starts = np.searchsorted(od_of_minor[order], np.arange(od_of_minor.max() + 2))
    rows_all, cols_all, data_all, bounds = [], [], [], []
    col0 = 0
    s2 = np.sqrt(2)
    for q in active_ods:
        rows = order[starts[q]:starts[q + 1]]
        used = rows[x0m[rows] > EPS_USED]
        if len(used) < 2:
            continue
        by_cost = used[np.argsort(c_minor[used])]
        cheap = by_cost[0]
        costly_list = by_cost[::-1][:per_od]
        for costly in costly_list:
            if costly == cheap:
                continue
            rows_all.extend([int(cheap), int(costly)])
            cols_all.extend([col0, col0])
            data_all.extend([1 / s2, -1 / s2])
            # alpha > 0 moves flow costly -> cheap; bounded by available anchor flow
            bounds.append((-s2 * float(x0m[cheap]), s2 * float(x0m[costly])))
            col0 += 1
    A = sp.csr_matrix((data_all, (rows_all, cols_all)),
                      shape=(len(od_of_minor), col0))
    return A, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="K10", choices=list(CASES))
    ap.add_argument("--scenario", default="d05")
    ap.add_argument("--outers", type=int, default=8)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--arms",
                    default="hyb500x1,hyb2000x1,hyb5000x1,hyb2000x2,dyn2000x1")
    a = ap.parse_args()
    dsdir, pool, tau = CASES[a.case]
    tag = f"{a.case}_{a.scenario}"
    rows_out = []
    K.banner(f"exp11 HYBRID origin backbone + active-OD exchange atoms ({tag})\n"
             f"  references: origin-light plateau 0.01132 (exp10b), "
             f"full-space 0.00069 (exp08)")

    P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
    if a.scenario.startswith("d") and a.scenario != "base":
        delta = int(a.scenario[1:]) / 100.0
        rng = np.random.default_rng(100)
        P["d"] = P["d"] * (1.0 + rng.uniform(-delta, delta, P["d"].size))
    x_start = ca.project_feasible(np.maximum(P["x0"], 1e-3), P["A"], P["d"], P["p2od"])
    major = ca.split_major_minor(P, tau=tau)
    P2 = dict(P)
    P2["x0"] = x_start
    Cb0 = ca.build_compressed(P2, major, 6, weight_flows=x_start)
    ev = CommonEvaluator(P)

    minor_idx = np.where(Cb0["minor"])[0]
    B2 = P["B"][Cb0["minor"]]
    A2 = P["A"][:, Cb0["minor"]]
    od_of_minor = P["p2od"][minor_idx]
    od_o = origins_of_kept_ods(P)
    origin_of_minor = od_o[od_of_minor]

    def anchored(x_anchor):
        x0m = np.maximum(x_anchor, 0.0)[minor_idx]
        d_eff = P["d"] - np.asarray(A2 @ x0m).flatten()
        v_base = P.get("v0", 0.0) + np.asarray(B2.T @ x0m).flatten()
        return x0m, d_eff, v_base

    x0m0, d_eff0, v_base0 = anchored(x_start)
    wgt = np.sqrt(np.maximum(x0m0, 0.0)) + 1e-9
    U_O = build_origin_blocks(B2, wgt, origin_of_minor, 0.05, 4, P["m"])
    R_O = U_O.shape[1]
    Gq0, c0 = od_gap_and_costs(P, x_start)
    print(f"  backbone R={R_O:,}; initial sum G_q = {Gq0.sum():,.0f}", flush=True)

    def make_hybrid_cb(x0m, d_eff, v_base, A_atoms):
        U = sp.hstack([U_O, A_atoms]).tocsr()
        return dict(B1=Cb0["B1"], A1=Cb0["A1"], U=U, D=(B2.T @ U), M=(A2 @ U),
                    d_eff=d_eff, v_base=v_base, x0m=x0m, major=Cb0["major"],
                    minor=Cb0["minor"], r=U.shape[1], svd_time=0.0)

    def run_arm(name, B, per_od, dynamic):
        Gs, cpu_tot = [], 0.0
        t_targets = {1e-2: None, 7e-3: None, 5e-3: None}
        if not dynamic:
            active = np.argsort(Gq0)[::-1][:B]
            A_atoms, bnd = build_atoms(od_of_minor, x0m0, c0[minor_idx], active, per_od)
            Cb = make_hybrid_cb(x0m0, d_eff0, v_base0, A_atoms)
            st = HybridStepper(P, Cb, x_start, C.TOL, a.inner,
                               [(None, None)] * R_O + bnd)
            print(f"  {name}: atoms={A_atoms.shape[1]:,} (requested B={B} x{per_od})",
                  flush=True)
            for k in range(1, a.outers + 1):
                rep = st.step()
                x_now = st.export_x()
                m = ev(x_now)
                Gs.append(m["G"])
                cpu_tot += rep["cpu_s"]
                for tgt in t_targets:
                    if t_targets[tgt] is None and m["G"] <= tgt:
                        t_targets[tgt] = round(cpu_tot, 1)
                u = Cb["x0m"] + np.asarray(Cb["U"] @ st.z).flatten()
                fb = float(np.mean(u <= 1e-9))
                print(f"  {name:11s} k={k}  G={m['G']:.5f}  od={m['r_od']:.5f}  "
                      f"cpu={rep['cpu_s']:6.1f}s  nit={rep['nit']:>4}  "
                      f"frac0={fb:.3f}", flush=True)
                rows_out.append(dict(arm=name, k=k, G=round(m["G"], 6),
                                     r_od=round(m["r_od"], 6), atoms=A_atoms.shape[1],
                                     cpu_s=round(rep["cpu_s"], 1), nit=rep["nit"],
                                     frac0=round(fb, 4)))
                K.write(f"exp11_hybrid_{tag}", rows_out)
                if st.done:
                    break
        else:
            # dynamic exchange pricing: re-anchor + reprice every outer; lam, rho carry
            x_now = x_start.copy()
            lam = np.zeros(P["n_od"])
            rho = 100.0
            prev = np.inf
            for k in range(1, a.outers + 1):
                x0m, d_eff, v_base = anchored(x_now)
                Gq, c = od_gap_and_costs(P, x_now)
                active = np.argsort(Gq)[::-1][:B]
                A_atoms, bnd = build_atoms(od_of_minor, x0m, c[minor_idx],
                                           active, per_od)
                Cb = make_hybrid_cb(x0m, d_eff, v_base, A_atoms)
                st = HybridStepper(P, Cb, np.maximum(x_now, 0.0), C.TOL, a.inner,
                                   [(None, None)] * R_O + bnd)
                st.lam, st.rho, st.prev = lam.copy(), rho, prev
                rep = st.step()
                lam, rho, prev = st.lam.copy(), st.rho, st.prev
                x_now = st.export_x()
                m = ev(x_now)
                Gs.append(m["G"])
                cpu_tot += rep["cpu_s"]
                for tgt in t_targets:
                    if t_targets[tgt] is None and m["G"] <= tgt:
                        t_targets[tgt] = round(cpu_tot, 1)
                print(f"  {name:11s} k={k}  G={m['G']:.5f}  od={m['r_od']:.5f}  "
                      f"cpu={rep['cpu_s']:6.1f}s  nit={rep['nit']:>4}  "
                      f"atoms={A_atoms.shape[1]:,}", flush=True)
                rows_out.append(dict(arm=name, k=k, G=round(m["G"], 6),
                                     r_od=round(m["r_od"], 6), atoms=A_atoms.shape[1],
                                     cpu_s=round(rep["cpu_s"], 1), nit=rep["nit"]))
                K.write(f"exp11_hybrid_{tag}", rows_out)
        print(f"  {name:11s} PLATEAU ~ {min(Gs):.5f}  total cpu {cpu_tot:.0f}s  "
              f"t@1e-2={t_targets[1e-2]}  t@7e-3={t_targets[7e-3]}  "
              f"t@5e-3={t_targets[5e-3]}\n", flush=True)
        rows_out.append(dict(arm=name, k="min", G=round(min(Gs), 6),
                             cpu_s=round(cpu_tot, 1),
                             t_1em2=t_targets[1e-2], t_7em3=t_targets[7e-3],
                             t_5em3=t_targets[5e-3]))
        K.write(f"exp11_hybrid_{tag}", rows_out)

    for arm in [s.strip() for s in a.arms.split(",") if s.strip()]:
        if arm.startswith("hyb"):
            B, per = arm[3:].split("x")
            run_arm(arm, int(B), int(per), dynamic=False)
        elif arm.startswith("dyn"):
            B, per = arm[3:].split("x")
            run_arm(arm, int(B), int(per), dynamic=True)

    print(f"wrote {C.RESULTS / f'exp11_hybrid_{tag}.csv'}\n[exp11] DONE", flush=True)


if __name__ == "__main__":
    main()
