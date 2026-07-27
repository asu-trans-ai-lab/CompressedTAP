"""Experiment 8 -- Full-Compressed LOCKSTEP diagnostic runner (outer-iteration layer).

Motivation (2026-07-27): completed-run totals cannot distinguish
  (B) a wrong compressed descent direction,
  (C) objective falling while the equilibrium gap rises,
  (D) equal per-step progress but uneconomic per-step cost (decoder/OD overhead),
  (E) better per-step progress that the implementation fails to monetise.
exp05/exp07 only produced totals. This runner advances the full and the compressed solver
ONE ALM outer iteration at a time under a single controller, from a BYTE-IDENTICAL
iteration 0, and scores both after every step with a third, independently implemented
full-space evaluator. Nothing is inferred from wall totals.

Identical iteration 0 (checked, not assumed):
    x_start = project_feasible(max(x0, 1e-3))          # the full solver's own start
    full:        x^0   = x_start
    compressed:  w0    = x_start[minor],  z^0 = 0,  y^0 = x_start[major]
                 (d_eff, v_base rebuilt from THIS w0, so w0 + U z^0 == x_start[minor] exactly)
    lam^0 = 0, rho^0 = 100, same demand, same pool, same BPR, same tolerances.
The runner prints max|x_f^0 - x_c^0|, objective diff, link-flow diff, OD-residual diff and
ABORTS unless they are at machine precision.

Deviations from the certified solvers, both deliberate and disclosed:
  * y^0 = x_start[major] exactly (the certified compressed solver clips at 1e-3; clipping
    would break the identical-start requirement);
  * the basis offset is x_start[minor], not the pool nominal flow (same requirement);
  * one outer iteration per call -- the inner L-BFGS-B call, its objective/gradient code,
    the multiplier updates, the rho schedule and the per-side break semantics are copied
    verbatim from compressed_assignment.solve_full / solve_compressed (hard regime).

Common evaluator (independent -- calls NEITHER solver's gap code):
    on the raw exported/decoded path-flow vector x (clipped at 0 for link loading;
    min(x_raw) reported separately):
      Z            Beckmann network objective
      r_OD         sum_q |sum_p x_p - d_q| / sum_q d_q     and the max-form
      G            sum_q sum_p x_p (c_p - c_q^min) / sum_q d_q c_q^min   (pool-min costs)
      r_PG         ||x - Pi_X(x - alpha c)|| / max(1, ||x||),  Pi_X = per-OD Euclidean
                   projection onto {x >= 0, sum_p x_p = d_q} (v3_metrics.convert_euclid),
                   alpha = 1e-2 (fixed, stated; comparative use only)
    plus the ALM decomposition per side: Z, lam'r, 0.5 rho|r|^2, hinge term -- never one
    blended "cost".

Per-step CPU is split by kernel inside each side's objective/gradient callback:
    link aggregation | BPR (t, beckmann) | major gradient | latent gradient
    | minor decode + hinge (compressed only) | OD residual
so case D/E is decided by measurement, not by the wall total.

Checkpoints: every synchronized step k saves both sides' full state to
<TMP>/lockstep/<tag>/ck_XXX.npz, so the first diverging outer iteration can be replayed
later at inner-step granularity (layer 2, separate driver).

    python drivers/exp08_lockstep.py [--case K10] [--scenario d05] [--outers 10] [--rank 6]
    python drivers/exp08_lockstep.py --case V2 --outers 2          # fast self-test
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C

import compressed_assignment as ca
import v3_metrics as M

ALPHA_PG = 1e-2
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "lockstep"

CASES = {  # name -> (dsdir, pool file, tau)
    "V2":  (C.V2 / "sketch", "pool.csv", 4.54),
    "E0":  (C.DATA / "03_chicago_sketch", "path_pool_E0_baseline.csv", 4.54),
    "K10": (C.DATA / "03_chicago_sketch", "path_pool_K10.csv", 4.54),
    "K15": (C.DATA / "03_chicago_sketch", "path_pool_K15.csv", 4.54),
}


# ------------------------------------------------------------------ evaluator
class CommonEvaluator:
    """Independent full-space scorer. Implements its own gap/residual code on the raw
    path-flow vector; does not call pool_relgap or any solver internals."""

    def __init__(self, P):
        self.B, self.A, self.d, self.p2od = P["B"], P["A"], P["d"], P["p2od"]
        self.bpr, self.v0, self.n_od = P["bpr"], P.get("v0", 0.0), P["n_od"]

    def __call__(self, x_raw):
        x = np.maximum(x_raw, 0.0)
        v = self.v0 + np.asarray(self.B.T @ x).flatten()
        Z = float(self.bpr.beckmann(v))
        tl = self.bpr.t(v)
        c = np.asarray(self.B @ tl).flatten()
        odsum = np.bincount(self.p2od, weights=x, minlength=self.n_od)
        res = odsum - self.d
        r_od = float(np.abs(res).sum() / self.d.sum())
        r_od_max = float(np.max(np.abs(res) / np.maximum(self.d, 1e-9)))
        ordr = np.lexsort((c, self.p2od))
        grp = np.searchsorted(self.p2od[ordr], np.arange(self.n_od))
        cmin = c[ordr[grp]]
        denom = float(self.d @ cmin)
        G = (float(c @ x) - float(odsum @ cmin)) / max(denom, 1e-12)
        xe = M.convert_euclid(x - ALPHA_PG * c, self.d, self.p2od)
        r_pg = float(np.linalg.norm(x - xe) / max(1.0, np.linalg.norm(x)))
        return dict(Z=Z, G=G, r_od=r_od, r_od_max=r_od_max, r_pg=r_pg,
                    min_x_raw=float(np.min(x_raw)), v=v)


# ------------------------------------------------------------------ steppers
class FullStepper:
    """One ALM outer iteration of compressed_assignment.solve_full per step().
    fg is copied verbatim; kernel times are accumulated around the same expressions."""

    def __init__(self, P, x_start, tol, maxiter_inner):
        self.P, self.tol, self.maxit = P, tol, maxiter_inner
        self.x = x_start.copy()
        self.lam = np.zeros(P["n_od"])
        self.rho, self.prev = 100.0, np.inf
        self.done = False

    def step(self):
        P = self.P
        B, A, d, bpr = P["B"], P["A"], P["d"], P["bpr"]
        vbg = P.get("v0", 0.0)
        lam, rho = self.lam, self.rho
        kt = dict(link=0.0, bpr=0.0, grad=0.0, od=0.0)

        def fg(xv):
            t1 = time.perf_counter()
            v = vbg + np.asarray(B.T @ xv).flatten()
            t2 = time.perf_counter()
            r = np.asarray(A @ xv).flatten() - d
            t3 = time.perf_counter()
            f = bpr.beckmann(v) + lam @ r + 0.5 * rho * float(r @ r)
            gv = bpr.t(v)
            t4 = time.perf_counter()
            g = (np.asarray(B @ gv).flatten()
                 + np.asarray(A.T @ (lam + rho * r)).flatten())
            t5 = time.perf_counter()
            kt["link"] += t2 - t1
            kt["od"] += t3 - t2
            kt["bpr"] += t4 - t3
            kt["grad"] += t5 - t4
            return f, g

        t0 = time.perf_counter()
        res = minimize(lambda xv: fg(xv)[0], self.x, jac=lambda xv: fg(xv)[1],
                       method="L-BFGS-B", bounds=[(0, None)] * P["n"],
                       options={"maxiter": self.maxit, "gtol": self.tol * 0.1})
        cpu = time.perf_counter() - t0
        self.x = res.x
        r = np.asarray(A @ self.x).flatten() - d
        cons = float(np.linalg.norm(r))
        rep = dict(cpu_s=cpu, nit=int(res.nit), aug=float(res.fun), cons=cons,
                   lin_term=float(lam @ r), quad_term=float(0.5 * rho * (r @ r)),
                   hinge_term=0.0, rho=rho, lam_norm=float(np.linalg.norm(lam)), kt=kt)
        # verbatim break-before-update semantics of solve_full
        if cons < self.tol:
            self.done = True
        else:
            self.lam = lam + rho * r
            if cons > 0.25 * self.prev:
                self.rho = min(rho * 2, 1e6)
            self.prev = cons
        return rep

    def export_x(self):
        return self.x.copy()


class CompStepper:
    """One ALM outer iteration of solve_compressed (regime='hard') per step().
    fg copied verbatim (weighted-basis campaign path); kernel-timed."""

    def __init__(self, P, Cb, x_start, tol, maxiter_inner, c2=1e3):
        self.P, self.C, self.tol, self.maxit, self.c2 = P, Cb, tol, maxiter_inner, c2
        self.s_dim = int(Cb["major"].sum())
        self.r = Cb["r"]
        self.y = x_start[Cb["major"]].copy()      # NOT clipped: identical-start requirement
        self.z = np.zeros(self.r)
        self.lam = np.zeros(P["n_od"])
        self.mu = np.zeros(len(Cb["x0m"]))
        self.rho, self.prev = 100.0, np.inf
        self.done = False

    def step(self):
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
            return f, np.concatenate([gy, gz])

        bounds = [(0, None)] * s_dim + [(None, None)] * r
        t0 = time.perf_counter()
        res = minimize(lambda w: fg(w)[0], np.concatenate([self.y, self.z]),
                       jac=lambda w: fg(w)[1], method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.maxit, "gtol": self.tol * 0.1})
        cpu = time.perf_counter() - t0
        self.y, self.z = res.x[:s_dim], res.x[s_dim:]
        rr = (np.asarray(A1 @ self.y).flatten() + Mm @ self.z) - d_eff
        cons = float(np.linalg.norm(rr))
        u_full = x0m + U @ self.z
        Rplus = float(np.linalg.norm(np.minimum(u_full, 0.0)))
        phi_end = np.maximum(0.0, mu - c2 * u_full)
        rep = dict(cpu_s=cpu, nit=int(res.nit), aug=float(res.fun), cons=cons,
                   Rplus=Rplus, lin_term=float(lam @ rr),
                   quad_term=float(0.5 * rho * (rr @ rr)),
                   hinge_term=float(np.sum(phi_end ** 2 - mu ** 2)) / (2 * c2),
                   rho=rho, lam_norm=float(np.linalg.norm(lam)),
                   mu_norm=float(np.linalg.norm(mu)), kt=kt)
        # verbatim update-before-break semantics of solve_compressed
        self.mu = np.maximum(0.0, mu - c2 * u_full)
        self.lam = lam + rho * rr
        if cons < self.tol and Rplus < 1e-3:
            self.done = True
        else:
            if cons > 0.25 * self.prev:
                self.rho = min(rho * 2, 1e6)
            self.prev = cons
        return rep

    def export_x(self):
        x = np.zeros(self.P["n"])
        x[self.C["major"]] = self.y
        x[self.C["minor"]] = self.C["x0m"] + self.C["U"] @ self.z
        return x


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="K10", choices=list(CASES))
    ap.add_argument("--scenario", default="d05",
                    help="dNN = demand +/-NN%% (seed 100, exp07 convention); base = none")
    ap.add_argument("--outers", type=int, default=10)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--inner", type=int, default=200)
    a = ap.parse_args()
    dsdir, pool, tau = CASES[a.case]
    tag = f"{a.case}_{a.scenario}_r{a.rank}"
    ckdir = SCRATCH / tag
    ckdir.mkdir(parents=True, exist_ok=True)
    K.banner(f"exp08 LOCKSTEP {tag}: one outer iteration per side per step, "
             f"common evaluator, kernel timers")

    P = ca.load_problem(str(dsdir), pool, multi_path_only=True)
    if a.scenario.startswith("d") and a.scenario != "base":
        delta = int(a.scenario[1:]) / 100.0
        rng = np.random.default_rng(100)              # exp07 d05 convention
        P["d"] = P["d"] * (1.0 + rng.uniform(-delta, delta, P["d"].size))
    print(f"  n={P['n']:,} od={P['n_od']:,} links={P['bpr'].t0.size:,} "
          f"scenario={a.scenario}", flush=True)

    # ---- identical iteration 0
    x_start = ca.project_feasible(np.maximum(P["x0"], 1e-3), P["A"], P["d"], P["p2od"])
    major = ca.split_major_minor(P, tau=tau)
    P2 = dict(P)
    P2["x0"] = x_start                                # offset AND weights from x_start
    Cb = ca.build_compressed(P2, major, a.rank, weight_flows=x_start)
    red = 100.0 * (P["n"] - int(major.sum()) - Cb["r"]) / P["n"]
    print(f"  majors={int(major.sum()):,} r={Cb['r']} reduction={red:.1f}% "
          f"svd={Cb['svd_time']:.2f}s", flush=True)

    ev = CommonEvaluator(P)
    full = FullStepper(P, x_start, C.TOL, a.inner)
    comp = CompStepper(P, Cb, x_start, C.TOL, a.inner)

    xf0, xc0 = full.export_x(), comp.export_x()
    mf0, mc0 = ev(xf0), ev(xc0)
    gate = dict(
        max_abs_difference=float(np.max(np.abs(xf0 - xc0))),
        OD_conservation_difference=abs(mf0["r_od"] - mc0["r_od"]),
        link_flow_difference=float(np.max(np.abs(mf0["v"] - mc0["v"]))),
        objective_difference=abs(mf0["Z"] - mc0["Z"]))
    print("\n  ITERATION-0 IDENTITY GATE")
    for k, v in gate.items():
        print(f"    {k:28s} {v:.3e}")
    assert gate["max_abs_difference"] < 1e-9, "rabbits do not start at the same point"
    assert gate["objective_difference"] < 1e-6 * max(1.0, mf0["Z"]), "objective mismatch"
    print("    PASS -- both solvers start from the same full-space point\n", flush=True)

    rows = [dict(k=0, side="both", Z_full=mf0["Z"], Z_comp=mc0["Z"], G_full=mf0["G"],
                 G_comp=mc0["G"], pg_full=mf0["r_pg"], pg_comp=mc0["r_pg"],
                 od_full=mf0["r_od"], od_comp=mc0["r_od"])]
    K.write(f"exp08_lockstep_{tag}", rows)
    Zf_prev, Zc_prev = mf0["Z"], mc0["Z"]

    hdr = (f"{'k':>2s} | {'Z_full':>14s} {'Z_comp':>14s} | {'G_full':>9s} {'G_comp':>9s} | "
           f"{'PG_f':>8s} {'PG_c':>8s} | {'od_f':>8s} {'od_c':>8s} | "
           f"{'cpu_f':>6s} {'cpu_c':>6s} | {'dZ/s_f':>8s} {'dZ/s_c':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for k in range(1, a.outers + 1):
        rf = full.step() if not full.done else None
        rc = comp.step() if not comp.done else None
        xf, xc = full.export_x(), comp.export_x()
        mf, mc = ev(xf), ev(xc)
        rZf = (Zf_prev - mf["Z"]) / rf["cpu_s"] if rf else 0.0
        rZc = (Zc_prev - mc["Z"]) / rc["cpu_s"] if rc else 0.0
        print(f"{k:2d} | {mf['Z']:14.1f} {mc['Z']:14.1f} | {mf['G']:9.5f} {mc['G']:9.5f} | "
              f"{mf['r_pg']:8.5f} {mc['r_pg']:8.5f} | {mf['r_od']:8.5f} {mc['r_od']:8.5f} | "
              f"{(rf or {}).get('cpu_s', 0):6.1f} {(rc or {}).get('cpu_s', 0):6.1f} | "
              f"{rZf:8.1f} {rZc:8.1f}"
              + ("  [full done]" if full.done else "")
              + ("  [comp done]" if comp.done else ""), flush=True)
        row = dict(k=k, Z_full=mf["Z"], Z_comp=mc["Z"], G_full=mf["G"], G_comp=mc["G"],
                   pg_full=mf["r_pg"], pg_comp=mc["r_pg"],
                   od_full=mf["r_od"], od_comp=mc["r_od"],
                   od_max_full=mf["r_od_max"], od_max_comp=mc["r_od_max"],
                   min_x_comp=mc["min_x_raw"], dZ_per_s_full=round(rZf, 2),
                   dZ_per_s_comp=round(rZc, 2))
        for side, rep in (("full", rf), ("comp", rc)):
            if rep is None:
                continue
            row[f"cpu_{side}"] = round(rep["cpu_s"], 3)
            row[f"nit_{side}"] = rep["nit"]
            row[f"aug_{side}"] = rep["aug"]
            row[f"cons_{side}"] = rep["cons"]
            row[f"lin_{side}"] = rep["lin_term"]
            row[f"quad_{side}"] = rep["quad_term"]
            row[f"hinge_{side}"] = rep["hinge_term"]
            row[f"rho_{side}"] = rep["rho"]
            row[f"lam_norm_{side}"] = round(rep["lam_norm"], 3)
            for kn, tv in rep["kt"].items():
                row[f"t_{side}_{kn}"] = round(tv, 3)
        if rc:
            row["Rplus_comp"] = rc["Rplus"]
            row["mu_norm_comp"] = round(rc["mu_norm"], 3)
        rows.append(row)
        K.write(f"exp08_lockstep_{tag}", rows)
        np.savez_compressed(ckdir / f"ck_{k:03d}.npz",
                            x_full=xf, lam_f=full.lam, rho_f=full.rho,
                            y=comp.y, z=comp.z, lam_c=comp.lam, mu=comp.mu,
                            rho_c=comp.rho)
        Zf_prev, Zc_prev = mf["Z"], mc["Z"]
        if full.done and comp.done:
            break

    # kernel CPU summary over the whole run
    print("\n  KERNEL CPU TOTALS (s)")
    for side in ("full", "comp"):
        ks = {}
        for row in rows[1:]:
            for key, v in row.items():
                if key.startswith(f"t_{side}_"):
                    ks[key[len(f"t_{side}_"):]] = ks.get(key[len(f"t_{side}_"):], 0) + v
        tot = sum(ks.values())
        if tot:
            print(f"    {side:5s} " + "  ".join(
                f"{k}={v:.2f} ({100*v/tot:.0f}%)" for k, v in
                sorted(ks.items(), key=lambda kv: -kv[1])))
    print(f"\nwrote {C.RESULTS / f'exp08_lockstep_{tag}.csv'}\n"
          f"checkpoints in {ckdir}\n[exp08] DONE", flush=True)


if __name__ == "__main__":
    main()
