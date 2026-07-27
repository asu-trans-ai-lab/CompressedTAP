"""Experiment 10 -- STRUCTURED representations at the certified plateau (Experiment A).

exp08/exp09 certified that the global signed low-rank basis is the root cause of the
equilibrium-gap floor on K10-d05: at the r=6 plateau span(U_6) captures 1.26% of the
minor-block equilibrium-correcting gradient, the small aligned component is boundary-
blocked, and rank 6->20 barely moves the floor. The next question is REPRESENTATION CLASS,
not rank: the equilibrium correction is within-OD flow exchange, and a global latent
coordinate couples unrelated ODs, so a useful direction for one OD forces negative decoded
flows elsewhere and the hinge blocks the whole direction.

This driver evaluates candidate representation classes AT THE SAME PLATEAU STATE, by three
criteria (gradient capture, one-step G reduction from the identical iterate and ALM state,
boundary activity), plus per-kernel CPU:

  global      the incumbent: one global weighted-SVD basis (baseline; eta ~ 1.26%)
  origin-*    block-diagonal by ORIGIN: one small weighted-SVD basis per origin,
              r_o = clamp(ceil(alpha * sqrt(N_o)), 1, rmax) -- corrections for one origin
              cannot disturb another; two budgets (light/medium)
  odx1        ONE exchange atom per eligible OD: a_q = (e_j - e_j')/sqrt(2) between the
              two largest-flow minors; 1'a_q = 0, so OD conservation is automatic
  odxfull     the OD-local benchmark: an orthonormal (Helmert) basis of the full
              within-OD zero-sum minor subspace, k_q - 1 atoms per OD -- the upper bound
              on what OD-locality can achieve, with little compression left
  escape      one full-space ALM step from the same state (the exp09 C4 reference,
              recomputed in-process)

All representations are anchored at the plateau: w0' = max(x_C^k4, 0)[minor], z = 0, so
every one-step comparison starts from the identical full-space point with the compressed
side's (lam, rho). The hinge multiplier mu restarts at 0 for every representation
(including the global baseline, so the comparison is internally fair); this is disclosed.

Capture-ratio caveat: g_T is centered per OD over ALL of the OD's paths, so a within-minor
zero-sum basis cannot capture the per-OD mean component of the minor block -- that
component corresponds to major<->minor exchange, which the explicit majors handle through
the conservation constraint. odxfull's eta is therefore the ceiling for minor-only
locality, not 1.0 by construction.

    python drivers/exp10_structured.py [--case K10] [--scenario d05] [--branch-k 4]
                                       [--reps global,origin:0.05:4,origin:0.15:12,odx1,odxfull,escape]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K
import config as C

import compressed_assignment as ca
from exp08_lockstep import CASES, CommonEvaluator, FullStepper, CompStepper


def origins_of_kept_ods(P):
    """Origin zone per kept OD index, replicating load_problem's keep logic exactly."""
    pool = P["pool"]
    od_key = list(zip(pool.o_zone_id, pool.d_zone_id))
    od_unique = sorted(set(od_key))
    idx = {od: i for i, od in enumerate(od_unique)}
    p2od_full = np.array([idx[k] for k in od_key])
    cnt = np.bincount(p2od_full, minlength=len(od_unique))
    keep = cnt > 1
    od_o = np.array([o for o, _ in od_unique])[keep]
    assert len(od_o) == P["n_od"], "origin reconstruction does not match load_problem"
    return od_o


def helmert(k):
    """k x (k-1) orthonormal basis of the zero-sum subspace of R^k."""
    H = np.zeros((k, k - 1))
    for j in range(1, k):
        H[:j, j - 1] = 1.0 / np.sqrt(j * (j + 1))
        H[j, j - 1] = -j / np.sqrt(j * (j + 1))
    return H


def build_origin_blocks(B2, wgt, origin_of_minor, alpha, rmax, m_links):
    """Block-diagonal weighted-SVD basis, one block per origin. Returns sparse U."""
    rows_all, cols_all, data_all = [], [], []
    col0 = 0
    for o in np.unique(origin_of_minor):
        rows = np.where(origin_of_minor == o)[0]
        N = len(rows)
        r_o = int(max(1, min(rmax, np.ceil(alpha * np.sqrt(N)))))
        r_o = min(r_o, N - 1, m_links - 1)
        if r_o < 1:
            continue
        blk = sp.diags(wgt[rows]) @ B2[rows]
        try:
            Uo, s, _ = svds(blk.astype(float), k=r_o)
        except Exception:
            continue
        Uo = Uo[:, np.argsort(s)[::-1]]
        for j in range(Uo.shape[1]):
            rows_all.extend(rows.tolist())
            cols_all.extend([col0 + j] * N)
            data_all.extend(Uo[:, j].tolist())
        col0 += Uo.shape[1]
    return sp.csr_matrix((data_all, (rows_all, cols_all)),
                         shape=(B2.shape[0], col0))


def build_odx(od_of_minor, x0m, kper):
    """Exchange atoms per OD. kper=1: one atom (two largest-flow minors);
    kper=0: full Helmert basis of the within-OD zero-sum minor subspace."""
    order = np.argsort(od_of_minor, kind="stable")
    rows_all, cols_all, data_all = [], [], []
    col0 = 0
    i = 0
    n = len(od_of_minor)
    while i < n:
        j = i
        while j < n and od_of_minor[order[j]] == od_of_minor[order[i]]:
            j += 1
        rows = order[i:j]
        i = j
        if len(rows) < 2:
            continue
        if kper == 1:
            top = rows[np.argsort(x0m[rows])[::-1][:2]]
            rows_all.extend([int(top[0]), int(top[1])])
            cols_all.extend([col0, col0])
            data_all.extend([1 / np.sqrt(2), -1 / np.sqrt(2)])
            col0 += 1
        else:
            H = helmert(len(rows))
            for c in range(H.shape[1]):
                rows_all.extend(rows.tolist())
                cols_all.extend([col0 + c] * len(rows))
                data_all.extend(H[:, c].tolist())
            col0 += H.shape[1]
    return sp.csr_matrix((data_all, (rows_all, cols_all)), shape=(n, col0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="K10", choices=list(CASES))
    ap.add_argument("--scenario", default="d05")
    ap.add_argument("--branch-k", type=int, default=4)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--reps",
                    default="global,origin:0.05:4,origin:0.15:12,odx1,odxfull,escape")
    a = ap.parse_args()
    dsdir, pool, tau = CASES[a.case]
    tag = f"{a.case}_{a.scenario}"
    rows_out = []
    K.banner(f"exp10 STRUCTURED representations at the k={a.branch_k} plateau ({tag})")

    # ---- rebuild the certified configuration and walk to the plateau (comp side only)
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
    comp = CompStepper(P, Cb, x_start, C.TOL, a.inner)
    for k in range(1, a.branch_k + 1):
        comp.step()
        print(f"    walk k={k}  G_comp={ev(comp.export_x())['G']:.5f}", flush=True)
    x_c4 = comp.export_x()
    lam_c, rho_c = comp.lam.copy(), comp.rho
    xc = np.maximum(x_c4, 0.0)
    G0 = ev(xc)["G"]
    print(f"  plateau G = {G0:.5f}\n", flush=True)

    # ---- shared plateau objects
    minor_idx = np.where(Cb["minor"])[0]
    B2 = P["B"][Cb["minor"]]
    A2 = P["A"][:, Cb["minor"]]
    x0m_new = xc[minor_idx]
    d_eff = P["d"] - np.asarray(A2 @ x0m_new).flatten()
    v_base = P.get("v0", 0.0) + np.asarray(B2.T @ x0m_new).flatten()
    od_of_minor = P["p2od"][minor_idx]
    od_o = origins_of_kept_ods(P)
    origin_of_minor = od_o[od_of_minor]
    wgt = np.sqrt(np.maximum(x0m_new, 0.0)) + 1e-9

    # g_T at the plateau (same construction as exp09 C3)
    v = P.get("v0", 0.0) + np.asarray(P["B"].T @ xc).flatten()
    c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
    cnt = np.bincount(P["p2od"], minlength=P["n_od"]).astype(float)
    odmean = np.bincount(P["p2od"], weights=c, minlength=P["n_od"]) / np.maximum(cnt, 1)
    gT = c - odmean[P["p2od"]]
    g_min = gT[Cb["minor"]]
    n_gmin = float(np.linalg.norm(g_min) ** 2)

    def one_step(name, U):
        R = U.shape[1]
        eta = float(np.linalg.norm(U.T @ g_min) ** 2) / n_gmin
        t0 = time.perf_counter()
        D = (B2.T @ U)
        Mm = (A2 @ U)
        t_build = time.perf_counter() - t0
        Cb2 = dict(B1=Cb["B1"], A1=Cb["A1"], U=U, D=D, M=Mm, d_eff=d_eff,
                   v_base=v_base, x0m=x0m_new, major=Cb["major"],
                   minor=Cb["minor"], r=R, svd_time=0.0)
        st = CompStepper(P, Cb2, xc, C.TOL, a.inner)
        st.lam, st.rho = lam_c.copy(), rho_c
        rep = st.step()
        x_after = st.export_x()
        G1 = ev(x_after)["G"]
        u = x0m_new + np.asarray(U @ st.z).flatten()
        frac_b = float(np.mean(u <= 1e-9))
        Rplus = float(np.linalg.norm(np.minimum(u, 0.0)))
        print(f"  {name:16s} R={R:>7,}  eta_minor={eta:8.4f}  "
              f"G {G0:.5f} -> {G1:.5f}  cpu={rep['cpu_s']:6.1f}s  "
              f"decode={rep['kt']['decode_hinge']:6.1f}s  nit={rep['nit']:>4}  "
              f"frac(u<=0)={frac_b:.4f}", flush=True)
        rows_out.append(dict(rep=name, R=R, eta_minor=round(eta, 6),
                             G_before=round(G0, 6), G_after=round(G1, 6),
                             dG=round(G0 - G1, 6), cpu_s=round(rep["cpu_s"], 1),
                             build_s=round(t_build, 1),
                             decode_cpu_s=round(rep["kt"]["decode_hinge"], 1),
                             nit=rep["nit"], boundary_frac=round(frac_b, 4),
                             Rplus=round(Rplus, 4)))
        K.write(f"exp10_structured_{tag}", rows_out)

    for spec in [s.strip() for s in a.reps.split(",") if s.strip()]:
        if spec == "global":
            one_step("global_r%d" % Cb["r"], sp.csr_matrix(Cb["U"]))
        elif spec.startswith("origin"):
            _, al, rm = spec.split(":")
            t0 = time.perf_counter()
            U = build_origin_blocks(B2, wgt, origin_of_minor, float(al), int(rm), P["m"])
            print(f"  [origin blocks alpha={al} rmax={rm}: R={U.shape[1]:,} "
                  f"built in {time.perf_counter()-t0:.1f}s]", flush=True)
            one_step(f"origin_a{al}_r{rm}", U)
        elif spec == "odx1":
            one_step("odx1", build_odx(od_of_minor, x0m_new, 1))
        elif spec == "odxfull":
            one_step("odxfull", build_odx(od_of_minor, x0m_new, 0))
        elif spec == "escape":
            esc = FullStepper(P, xc, C.TOL, a.inner)
            esc.lam, esc.rho = lam_c.copy(), rho_c
            rep = esc.step()
            G1 = ev(esc.export_x())["G"]
            print(f"  {'escape_full':16s} R={P['n']:>7,}  eta_minor=  1.0000  "
                  f"G {G0:.5f} -> {G1:.5f}  cpu={rep['cpu_s']:6.1f}s", flush=True)
            rows_out.append(dict(rep="escape_full", R=P["n"], eta_minor=1.0,
                                 G_before=round(G0, 6), G_after=round(G1, 6),
                                 dG=round(G0 - G1, 6), cpu_s=round(rep["cpu_s"], 1)))
            K.write(f"exp10_structured_{tag}", rows_out)

    print(f"\nwrote {C.RESULTS / f'exp10_structured_{tag}.csv'}\n[exp10] DONE", flush=True)


if __name__ == "__main__":
    main()
