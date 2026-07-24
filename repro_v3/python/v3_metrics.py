"""v3 evaluation metrics — ONE implementation shared by every table.

Implements Eq. (metric) of the v3 manuscript:

    x^F = P_X(x~)                     common feasible conversion, OD pair by OD pair
    delta_F(%) = 100 * ||x^F - x~||_1 / max{1, ||x^F||_1}
    Gap_F(%)   = 100 * (f(v^F) - f(v^ref)) / f(v^ref)
    R^2        = 1 - SSE(v^F, v^ref) / SST(v^ref)

with v^F = B' x^F + v_0 and v^ref a consistently converged feasible reference.

TWO CONVERSION MAPS (spec ambiguity, resolved by reporting both)
----------------------------------------------------------------
The manuscript writes P_X, which conventionally denotes the Euclidean projection onto
X = {x >= 0, Ax = d}.  The certified pipeline's `project_feasible` instead performs a
multiplicative rescaling (clip to zero, then scale each OD block to meet its demand).
Both land in X but they are different maps, and delta_F -- an L1 distance to x~ -- takes
different values under each.  Every driver therefore records BOTH:

    conv = "euclid"   per-OD Euclidean projection onto the demand simplex (matches P_X)
    conv = "rescale"  clip-and-rescale (the certified pipeline's historical map)

`euclid` is the default and the one intended for the paper; `rescale` is carried as a
diagnostic so the choice is visible rather than buried.

Author decision 2026-07-19: all CPU columns are re-timed from fresh runs; no submitted
timings are reused, so no row mixes two campaigns.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- conversions
def project_simplex_block(y: np.ndarray, mass: float) -> np.ndarray:
    """Euclidean projection of y onto {u >= 0, sum(u) = mass} (Held-Wolfe/Duchi sort)."""
    if mass <= 0:
        return np.zeros_like(y)
    n = y.size
    if n == 1:
        return np.array([mass])
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - mass
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    if not cond.any():
        return np.full(n, mass / n)
    rho = int(np.nonzero(cond)[0][-1])
    theta = css[rho] / (rho + 1.0)
    return np.maximum(y - theta, 0.0)


def convert_euclid(x: np.ndarray, d: np.ndarray, p2od: np.ndarray,
                   od_slices=None) -> np.ndarray:
    """Per-OD Euclidean projection onto {x >= 0, Ax = d}.  p2od maps path -> OD index.

    Segmented and fully vectorised: one lexsort plus a segmented cumulative sum, with no
    Python loop over OD pairs.  This routine is called on every iteration of every
    operator and inside every certificate evaluation, so the loop version dominated the
    run time on instances with many OD pairs (see REMARKS.md R8).
    `od_slices` is accepted and ignored, for call-site compatibility.
    """
    x = np.asarray(x, dtype=float)
    p2od = np.asarray(p2od)
    n_od = len(d)
    # sort by (OD, descending value)
    order = np.lexsort((-x, p2od))
    xs = x[order]
    ods = p2od[order]
    # block starts / sizes in the sorted array
    starts = np.searchsorted(ods, np.arange(n_od))
    sizes = np.diff(np.append(starts, len(xs)))
    # within-block descending cumulative sum
    cs = np.cumsum(xs)
    base = np.zeros(n_od)
    nz = sizes > 0
    base[nz] = cs[starts[nz]] - xs[starts[nz]]          # sum strictly before each block
    seg_cs = cs - base[ods]
    rank = np.arange(len(xs)) - starts[ods] + 1          # 1-based position in block
    css = seg_cs - d[ods]
    cond = xs - css / rank > 0
    # last True per block = largest rank with cond
    rho = np.zeros(n_od, dtype=int)
    np.maximum.at(rho, ods[cond], rank[cond])
    theta = np.zeros(n_od)
    has = rho > 0
    if np.any(has):
        pos = starts[has] + rho[has] - 1
        theta[has] = css[pos] / rho[has]
    # blocks where no index satisfied the condition fall back to the uniform split
    if np.any(~has & nz):
        bad = np.nonzero(~has & nz)[0]
        theta[bad] = (seg_cs[starts[bad] + sizes[bad] - 1] - d[bad]) / sizes[bad]
    xf = np.maximum(x - theta[p2od], 0.0)

    # Numerical guard. seg_cs is formed by subtracting one global cumulative sum from
    # another; on very large pools those two quantities are large and nearly equal, so
    # cancellation can cost significant digits. Verify the result actually lands on the
    # demand constraint and fall back to the per-block routine if it does not. The check
    # is O(n) and, on every instance tested so far, never fires.
    got = np.zeros(n_od)
    np.add.at(got, p2od, xf)
    scale = np.maximum(np.abs(d), 1.0)
    if np.max(np.abs(got - d) / scale) > 1e-9:
        xf = np.empty_like(x)
        order2 = np.argsort(p2od, kind="stable")
        st = np.searchsorted(p2od[order2], np.arange(n_od))
        en = np.append(st[1:], len(order2))
        for i in range(n_od):
            idx = order2[st[i]:en[i]]
            if idx.size:
                xf[idx] = project_simplex_block(x[idx], float(d[i]))
    return xf


def convert_rescale(x: np.ndarray, A, d: np.ndarray, p2od: np.ndarray) -> np.ndarray:
    """Clip-and-rescale (the certified pipeline's `project_feasible`)."""
    s = np.asarray(A @ np.maximum(x, 0.0)).flatten()
    sc = np.divide(d, s, out=np.ones_like(d, dtype=float), where=s > 1e-12)
    return np.maximum(x, 0.0) * sc[p2od]


# --------------------------------------------------------------------------- metrics
def link_flow(P, xf: np.ndarray) -> np.ndarray:
    return P.get("v0", 0.0) + np.asarray(P["B"].T @ xf).flatten()


def link_r2(v: np.ndarray, vref: np.ndarray) -> float:
    sst = float(np.sum((vref - vref.mean()) ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - float(np.sum((v - vref) ** 2)) / sst


def evaluate(P, x_tilde: np.ndarray, obj_ref: float, v_ref: np.ndarray,
             conv: str = "euclid", od_slices=None) -> dict:
    """Full v3 metric set for one terminal path-flow vector.

    obj_ref / v_ref must come from the consistently converged feasible reference
    (the tau = 0 ALM run), evaluated through this same module.
    """
    x_tilde = np.asarray(x_tilde, dtype=float)
    if conv == "euclid":
        xf = convert_euclid(x_tilde, P["d"], P["p2od"], od_slices)
    elif conv == "rescale":
        xf = convert_rescale(x_tilde, P["A"], P["d"], P["p2od"])
    else:
        raise ValueError("conv must be 'euclid' or 'rescale'")

    l1 = float(np.sum(np.abs(xf - x_tilde)))
    denom = max(1.0, float(np.sum(np.abs(xf))))
    v = link_flow(P, xf)
    obj = float(P["bpr"].beckmann(v))
    demand_res = float(np.max(np.abs(np.asarray(P["A"] @ xf).flatten() - P["d"])))
    return {
        "conv": conv,
        "delta_F_pct": 100.0 * l1 / denom,
        "obj_feasible": obj,
        "Gap_F_pct": 100.0 * (obj - obj_ref) / obj_ref,
        "link_R2": link_r2(v, v_ref),
        "link_diff_pct": 100.0 * float(np.linalg.norm(v - v_ref))
                         / max(float(np.linalg.norm(v_ref)), 1e-15),
        "demand_residual_inf": demand_res,
        "min_path_flow": float(np.min(xf)),
    }


def evaluate_both(P, x_tilde, obj_ref, v_ref, od_slices=None) -> dict:
    """Both conversion maps; euclid keys unprefixed (paper), rescale keys prefixed."""
    e = evaluate(P, x_tilde, obj_ref, v_ref, "euclid", od_slices)
    r = evaluate(P, x_tilde, obj_ref, v_ref, "rescale")
    out = dict(e)
    for k, v in r.items():
        if k != "conv":
            out["rescale_" + k] = v
    return out


def raw_objective_diff_pct(P, x_tilde: np.ndarray, obj_ref: float) -> float:
    """Raw terminal difference (diagnostic only; may be negative if x~ is infeasible)."""
    v = link_flow(P, np.asarray(x_tilde, dtype=float))
    return 100.0 * (float(P["bpr"].beckmann(v)) - obj_ref) / obj_ref


def od_slice_index(P):
    """Precompute per-OD path index arrays once (large pools: reused across evaluations)."""
    p2od = P["p2od"]
    order = np.argsort(p2od, kind="stable")
    starts = np.searchsorted(p2od[order], np.arange(P["n_od"]))
    ends = np.append(starts[1:], len(order))
    return [order[s:e] for s, e in zip(starts, ends)]
