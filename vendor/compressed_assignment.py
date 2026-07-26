#!/usr/bin/env python3
"""
Layer 2+3 — Compressed assignment core + constraint-regime handler +
recovery diagnostics, operating on the standard dataset + path_pool.csv.

Formulation (paper notation):
    x = (y, w),  w ~ x0m + U_r z          (major explicit / minor latent)
    v = B1'y + B2'(x0m + U_r z)
    min f(v)  s.t.  A1 y + M z = d_eff,  y >= 0,  [x0m + U_r z >= 0]
with M = A2 U_r, d_eff = d - A2 x0m, D = B2' U_r.

Constraint regimes for the minor nonnegativity (C2 ablation):
    HARD    multiplier ALM on all n-s rows (the paper's original scheme)
    SOFT    fixed quadratic penalty, no multipliers
    SCREEN  penalty applied only to the currently-violated working set
    RECOVER drop the constraint; after solving, clip negatives and rescale
            per OD to restore Ax = d exactly; certify R+ and re-evaluate
FULL = L-BFGS-B ALM over all path variables (reference for time & quality).
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.optimize import minimize


# ---------------------------------------------------------------- data
class BPR:
    def __init__(self, link_df):
        self.cap = np.maximum(link_df.capacity.values.astype(float), 1.0)
        if 'vdf_free_speed_mph' in link_df.columns:
            spd = link_df.vdf_free_speed_mph.fillna(link_df.free_speed)
        else:
            spd = link_df.free_speed
        spd = np.maximum(spd.values.astype(float), 1.0)
        length = (link_df.vdf_length_mi if 'vdf_length_mi' in link_df.columns
                  else link_df.length).values.astype(float)
        self.t0 = np.maximum(60.0 * length / spd, 0.01)
        self.alpha = (link_df.vdf_alpha.fillna(0.15).values.astype(float)
                      if 'vdf_alpha' in link_df.columns
                      else np.full(len(link_df), 0.15))
        self.beta = (link_df.vdf_beta.fillna(4.0).values.astype(float)
                     if 'vdf_beta' in link_df.columns
                     else np.full(len(link_df), 4.0))

    def t(self, v):
        return self.t0 * (1 + self.alpha * (np.maximum(v, 0) / self.cap) ** self.beta)

    def dt(self, v):
        vr = np.maximum(v, 1e-10) / self.cap
        return self.t0 * self.alpha * self.beta / self.cap * vr ** (self.beta - 1)

    def beckmann(self, v):
        # The LINEAR term must use v, not max(v,0): with max(v,0) the objective is flat for
        # v<0 (derivative 0) while t(v) still returns t0, so the gradient contradicts the
        # function and L-BFGS-B's line search fails (ABNORMAL) the moment any link flow goes
        # negative. As written, d(beckmann)/dv == t(v) identically for ALL v. Feasible points
        # (v>=0) are numerically unchanged; only the extension to v<0 differs. This matters
        # for the anchor-substituted models, whose reduced incidence B[p]-B[a(w)] has -1
        # entries and so can probe v<0 during a line search.
        vp = np.maximum(v, 0)
        return float(np.sum(self.t0 * v + self.t0 * self.alpha * self.cap
                            / (self.beta + 1) * (vp / self.cap) ** (self.beta + 1)))


def load_problem(dataset_dir, pool_file='path_pool.csv', multi_path_only=False):
    """Load a GMNS instance.

    multi_path_only=False (default) preserves the original behaviour exactly: every path in the
    pool becomes a variable.

    multi_path_only=True restricts the optimisation to OD pairs that actually offer a CHOICE, and
    carries the rest as a constant link-flow background v0. An OD pair with a single path has no
    decision to make -- conservation fixes its flow -- so keeping it as a variable adds nothing but
    work, and it distorts the reported richness: on the submitted Chicago Sketch instance the full
    pool reads Kbar = 1.27 across 93,135 OD pairs, while the actual choice problem is Kbar = 2.45
    across 17,464. The manuscript's Table 1 (n = 42,774 / ell = 17,464) is the latter.

    This mirrors what the submitted solver does (compressed_tap.py builds v_0 from B_singleton and
    d_singleton). Dropping singletons WITHOUT the background would be wrong, not merely smaller:
    their flow still loads links, so omitting it changes every link cost and hence the equilibrium.
    P['v0'] is that background; any objective must add it to B.T @ x.
    """
    link = pd.read_csv(os.path.join(dataset_dir, 'link.csv'), low_memory=False)
    link_ids = link.link_id.values
    lid2idx = {lid: i for i, lid in enumerate(link_ids)}
    pool = pd.read_csv(os.path.join(dataset_dir, pool_file))
    dem = pd.read_csv(os.path.join(dataset_dir, 'demand.csv'))
    dem = dem.groupby(['o_zone_id', 'd_zone_id'], as_index=False).volume.sum()

    od_key = list(zip(pool.o_zone_id, pool.d_zone_id))
    od_unique = sorted(set(od_key))
    od2idx = {od: i for i, od in enumerate(od_unique)}
    p2od = np.array([od2idx[k] for k in od_key])

    rows, cols = [], []
    for p, seq in enumerate(pool.link_ids):
        for tok in str(seq).split(';'):
            if tok and int(tok) in lid2idx:
                rows.append(p)
                cols.append(lid2idx[int(tok)])
    n, m = len(pool), len(link)
    B = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, m))
    A = csr_matrix((np.ones(n), (p2od, np.arange(n))),
                   shape=(len(od_unique), n))

    dmap = {(r.o_zone_id, r.d_zone_id): r.volume for r in dem.itertuples()}
    # float64 explicitly: an integer demand column otherwise yields an int array, and
    # project_feasible's np.divide(..., out=np.ones_like(d)) then fails to cast.
    d = np.array([dmap.get(od, 0.0) for od in od_unique], dtype=float)
    covered = d.sum() / dem.volume.sum()
    x0 = np.nan_to_num(pool.volume_ref.values.astype(float), nan=0.0)
    x0 = np.maximum(x0, 0.0)
    v0 = np.zeros(m)
    if multi_path_only:
        cnt = np.bincount(p2od, minlength=len(od_unique))
        keep_od = cnt > 1
        keep_p = keep_od[p2od]
        if not keep_p.all():
            # singleton ODs carry demand, not decisions: their flow is fixed by conservation, so
            # it becomes a constant link load rather than a variable.
            x_single = np.zeros(n)
            x_single[~keep_p] = d[p2od[~keep_p]]
            v0 = np.asarray(B.T @ x_single).flatten()
            keep_idx = np.where(keep_p)[0]
            remap = -np.ones(len(od_unique), dtype=np.int64)
            remap[np.where(keep_od)[0]] = np.arange(int(keep_od.sum()))
            B, x0, p2od = B[keep_idx], x0[keep_idx], remap[p2od[keep_idx]]
            d, od_unique = d[keep_od], [od_unique[i] for i in np.where(keep_od)[0]]
            n = len(keep_idx)
            A = csr_matrix((np.ones(n), (p2od, np.arange(n))), shape=(len(od_unique), n))
    return {'B': B, 'A': A, 'd': d, 'x0': x0, 'p2od': p2od, 'v0': v0,
            'bpr': BPR(link), 'n': n, 'm': m, 'n_od': len(od_unique),
            'demand_coverage': float(covered), 'pool': pool}


def project_feasible(x, A, d, p2od):
    s = np.asarray(A @ np.maximum(x, 0)).flatten()
    sc = np.divide(d, s, out=np.ones_like(d), where=s > 1e-12)
    return np.maximum(x, 0) * sc[p2od]


def pool_relgap(P, x):
    """In-pool relative gap of the feasibility-projected iterate — the same
    convergence certificate FW/GP/RSD report (for Xie-style gap-vs-time
    figures on a common metric). Link costs include the singleton background
    P['v0'] (zero unless the instance was loaded multi_path_only)."""
    B, d, p2od = P['B'], P['d'], P['p2od']
    xf = project_feasible(x, P['A'], d, p2od)
    tl = P['bpr'].t(P.get('v0', 0.0) + np.asarray(B.T @ xf).flatten())
    c = np.asarray(B @ tl).flatten()
    ordr = np.lexsort((c, p2od))
    grp = np.searchsorted(p2od[ordr], np.arange(P['n_od']))
    pick = ordr[grp]
    cx = float(c @ xf)
    cy = float(np.sum(d * c[pick]))
    return (cx - cy) / max(cy, 1e-12)


# ---------------------------------------------------------------- FULL
def solve_full(P, tol=1e-4, max_outer=40, maxiter_inner=200, verbose=False,
               track_pool_gap=False, max_seconds=None):
    # max_seconds (added 2026-07-19, purely additive; default None = original behavior):
    # a wall-clock guard checked at each OUTER iteration boundary, so a large-network run
    # cannot hang the multi-threshold campaign. It never interrupts an inner solve, so the
    # returned point is always a completed outer iterate.
    B, A, d, bpr = P['B'], P['A'], P['d'], P['bpr']
    n, n_od = P['n'], P['n_od']
    vbg = P.get('v0', 0.0)                  # singleton background link flow (0 unless multi_path_only)
    x = project_feasible(np.maximum(P['x0'], 1e-3), A, d, P['p2od'])
    lam = np.zeros(n_od)
    rho, prev = 100.0, np.inf
    t0 = time.time()
    nit_tot = 0
    history = []
    for outer in range(max_outer):
        if max_seconds is not None and (time.time() - t0) > max_seconds:
            break
        def fg(xv):
            v = vbg + np.asarray(B.T @ xv).flatten()
            r = np.asarray(A @ xv).flatten() - d
            f = bpr.beckmann(v) + lam @ r + 0.5 * rho * float(r @ r)
            g = (np.asarray(B @ bpr.t(v)).flatten()
                 + np.asarray(A.T @ (lam + rho * r)).flatten())
            return f, g
        res = minimize(lambda xv: fg(xv)[0], x, jac=lambda xv: fg(xv)[1],
                       method='L-BFGS-B', bounds=[(0, None)] * n,
                       options={'maxiter': maxiter_inner, 'gtol': tol * 0.1})
        x = res.x
        nit_tot += res.nit
        r = np.asarray(A @ x).flatten() - d
        cons = float(np.linalg.norm(r))
        if track_pool_gap:
            history.append((time.time() - t0, pool_relgap(P, x)))
        if verbose:
            print(f"    full outer {outer+1} cons {cons:.5f} "
                  f"obj {bpr.beckmann(vbg + np.asarray(B.T@x).flatten()):,.0f}")
        if cons < tol:
            break
        lam += rho * r
        if cons > 0.25 * prev:
            rho = min(rho * 2, 1e6)
        prev = cons
    xf = project_feasible(x, A, d, P['p2od'])
    v = vbg + np.asarray(B.T @ xf).flatten()
    # 'x_raw' (added 2026-07-19, purely additive): the UNPROJECTED terminal iterate.
    # The v3 metrics need it for the raw objective difference, ||Ax-d||_inf, and delta_F.
    # No existing key or computation is changed.
    return {'x': xf, 'v': v, 'obj': bpr.beckmann(v), 'x_raw': x,
            'time_s': time.time() - t0, 'inner_iters': nit_tot,
            'cons': cons, 'history': history}


# ------------------------------------------------- compressed problem
def split_major_minor(P, tau=1e-3):
    """Major: per-OD max-x0 path + any path with x0 > tau. Minor: rest."""
    x0, p2od, n_od = P['x0'], P['p2od'], P['n_od']
    best = {}
    for p in range(P['n']):
        w = p2od[p]
        if w not in best or x0[p] > x0[best[w]]:
            best[w] = p
    major = np.zeros(P['n'], bool)
    major[list(best.values())] = True
    major |= x0 > tau
    return major


def build_compressed(P, major, r, weight_flows=None):
    """weight_flows (optional): path-flow vector for ROW-SCALED basis selection.

    The SVD is taken of S B2 with S = diag(sqrt(f_minor) + 1e-9), and the resulting left
    singular vectors are used DIRECTLY as the path-flow basis Phi_r. High-flow minor paths
    therefore dominate the factorization, which is the intent.

    NOTE (verified 2026-07-25, verify_weighted_basis.py): this is a row-scaling heuristic, NOT
    the minimizer of the flow-weighted reconstruction loss sum_p f_p ||.||^2. By Eckart-Young
    that minimizer is span(S^-1 U~) -- the singular vectors mapped back out of the scaled
    coordinates -- whereas this returns span(U~). Measured at r=20, the achievable weighted
    loss is 249.1 here vs 209.1 for S^-1 U~ and 227.9 for the plain SVD on Sioux Falls; the two
    subspaces are nearly orthogonal there (80.6% of minors have zero nominal flow) and close on
    Chicago Sketch (no zeros). The heuristic is retained because it improves the FEASIBLE
    OBJECTIVE GAP of the solved problem in 27 of 34 paired comparisons, which is the quantity
    that matters; do not describe it as reconstruction-optimal.

    Default None = the certified unweighted behavior, bit-identical."""
    B, A, x0 = P['B'], P['A'], P['x0']
    minor = ~major
    B1, B2 = B[major], B[minor]
    A1, A2 = A[:, major], A[:, minor]
    x0m = x0[minor]
    t0 = time.time()
    k = min(r, min(B2.shape) - 1)
    if weight_flows is not None:
        import scipy.sparse as _sp
        wgt = np.sqrt(np.maximum(np.asarray(weight_flows)[minor], 0.0)) + 1e-9
        U, s, Vt = svds((_sp.diags(wgt) @ B2).astype(float), k=k)
    else:
        U, s, Vt = svds(B2.astype(float), k=k)
    idx = np.argsort(s)[::-1]
    U = U[:, idx]
    svd_time = time.time() - t0
    D = (B2.T @ U)                       # m x r  (dense)
    M = (A2 @ U)                         # n_od x r (dense)
    d_eff = P['d'] - np.asarray(A2 @ x0m).flatten()
    # v_base carries every constant link load: the minor reference flow AND the singleton
    # background P['v0'] (zero unless the instance was loaded multi_path_only).
    v_base = P.get('v0', 0.0) + np.asarray(B2.T @ x0m).flatten()
    return {'B1': B1, 'A1': A1, 'U': U, 'D': np.asarray(D),
            'M': np.asarray(M), 'd_eff': d_eff, 'v_base': v_base,
            'x0m': x0m, 'major': major, 'minor': minor,
            's_spectrum': s[idx], 'svd_time': svd_time, 'r': k}


def solve_compressed(P, C, regime='hard', tol=1e-4, max_outer=30,
                     maxiter_inner=200, c2=1e3, mu_soft=1e3, verbose=False,
                     track_pool_gap=False, max_seconds=None):
    """Regimes: hard | soft | screen | recover.

    max_seconds (added 2026-07-19, purely additive; default None = original behavior):
    wall-clock guard checked at each outer-iteration boundary."""
    bpr = P['bpr']
    B1, A1, U, D, M = C['B1'], C['A1'], C['U'], C['D'], C['M']
    d_eff, v_base, x0m = C['d_eff'], C['v_base'], C['x0m']
    s_dim = int(C['major'].sum())
    r = C['r']
    n_minor = len(x0m)
    y = np.maximum(P['x0'][C['major']], 1e-3)
    z = np.zeros(r)
    lam = np.zeros(P['n_od'])
    mu = np.zeros(n_minor) if regime == 'hard' else None
    Wset = np.zeros(n_minor, bool)        # screen working set
    rho, prev = 100.0, np.inf
    t0 = time.time()
    nit_tot = 0
    history = []
    minor_matvec_rows = 0                 # dense-row work counter (C2 metric)

    use_pen = regime in ('hard', 'soft', 'screen')
    for outer in range(max_outer):
        if max_seconds is not None and (time.time() - t0) > max_seconds:
            break
        Uw = U if regime in ('hard', 'soft') else (U[Wset] if Wset.any() else None)
        x0w = x0m if regime in ('hard', 'soft') else (x0m[Wset] if Wset.any() else None)
        muw = (mu if regime == 'hard'
               else (np.zeros(int(Wset.sum())) if regime == 'screen' and Wset.any() else None))

        def fg(yz):
            nonlocal minor_matvec_rows
            yv, zv = yz[:s_dim], yz[s_dim:]
            v = v_base + np.asarray(B1.T @ yv).flatten() + D @ zv
            rr = (np.asarray(A1 @ yv).flatten() + M @ zv) - d_eff
            f = bpr.beckmann(v) + lam @ rr + 0.5 * rho * float(rr @ rr)
            gv = bpr.t(v)
            gy = np.asarray(B1 @ gv).flatten() + np.asarray(A1.T @ (lam + rho * rr)).flatten()
            gz = D.T @ gv + M.T @ (lam + rho * rr)
            if use_pen and Uw is not None and len(x0w):
                u = x0w + Uw @ zv
                minor_matvec_rows += len(x0w)
                if regime == 'hard':
                    phi = np.maximum(0.0, muw - c2 * u)
                    f += float(np.sum(phi**2 - muw**2)) / (2 * c2)
                    gz += -(Uw.T @ phi)
                else:
                    neg = np.minimum(u, 0.0)
                    f += 0.5 * mu_soft * float(neg @ neg)
                    gz += mu_soft * (Uw.T @ neg)
            return f, np.concatenate([gy, gz])

        bounds = [(0, None)] * s_dim + [(None, None)] * r
        res = minimize(lambda w: fg(w)[0], np.concatenate([y, z]),
                       jac=lambda w: fg(w)[1], method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': maxiter_inner, 'gtol': tol * 0.1})
        y, z = res.x[:s_dim], res.x[s_dim:]
        nit_tot += res.nit
        rr = (np.asarray(A1 @ y).flatten() + M @ z) - d_eff
        cons = float(np.linalg.norm(rr))
        u_full = x0m + U @ z
        viol = np.minimum(u_full, 0.0)
        Rplus = float(np.linalg.norm(viol))
        if track_pool_gap:
            xt = np.zeros(P['n'])
            xt[C['major']] = y
            xt[C['minor']] = u_full
            history.append((time.time() - t0, pool_relgap(P, xt)))
        if verbose:
            print(f"    [{regime}] outer {outer+1} cons {cons:.5f} "
                  f"R+ {Rplus:.2f} nit {res.nit}")
        if regime == 'hard':
            mu = np.maximum(0.0, mu - c2 * u_full)
        if regime == 'screen':
            new = (u_full < -1e-6) & (~Wset)
            Wset |= new
        lam += rho * rr
        if cons < tol and (Rplus < 1e-3 or regime == 'recover'
                           or (regime == 'screen' and not (u_full < -1e-6).any())):
            break
        if cons > 0.25 * prev:
            rho = min(rho * 2, 1e6)
        prev = cons

    solve_time = time.time() - t0
    # reconstruct + recovery certification
    x = np.zeros(P['n'])
    x[C['major']] = y
    x[C['minor']] = x0m + U @ z
    Rplus_final = float(np.linalg.norm(np.minimum(x[C['minor']], 0)))
    xf = project_feasible(x, P['A'], P['d'], P['p2od'])
    v = P.get('v0', 0.0) + np.asarray(P['B'].T @ xf).flatten()
    return {'x': xf, 'v': v, 'obj': bpr.beckmann(v), 'x_raw': x,
            'time_s': solve_time, 'svd_time': C['svd_time'],
            'inner_iters': nit_tot, 'cons': cons, 'history': history,
            'Rplus_before_recovery': Rplus_final,
            'screen_set_size': int(Wset.sum()) if regime == 'screen' else None,
            'minor_penalty_row_work': int(minor_matvec_rows)}


def compare(sol, ref, bpr, link_percentiles=False):
    dv = sol['v'] - ref['v']
    tt = bpr.t(sol['v'])
    tt_ref = bpr.t(ref['v'])
    out = {
        'obj_gap_pct': round((sol['obj'] - ref['obj']) / ref['obj'] * 100, 4),
        'E_v': round(float(np.linalg.norm(dv) / np.linalg.norm(ref['v'])), 5),
        'E_t': round(float(np.linalg.norm(tt - tt_ref)
                           / np.linalg.norm(tt_ref)), 5),
    }
    if link_percentiles:
        # R1-3: link-level error DISTRIBUTION (relative, on meaningful links)
        vref = np.maximum(np.abs(ref['v']), 1.0)
        rel_v = np.abs(dv) / vref
        rel_t = np.abs(tt - tt_ref) / np.maximum(tt_ref, 1e-9)
        for tag, e in (('v', rel_v), ('t', rel_t)):
            for p in (50, 90, 99):
                out[f'link_{tag}_err_p{p}'] = round(
                    float(np.percentile(e, p)), 6)
            out[f'link_{tag}_err_max'] = round(float(e.max()), 6)
    return out
