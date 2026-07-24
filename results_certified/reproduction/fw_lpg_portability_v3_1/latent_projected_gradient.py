"""Latent Projected Gradient on OD-Demand Simplices (LPG).

Same representation layer as latent Frank--Wolfe: atoms A, latent
coordinates y with per-OD equality simplices {y >= 0, 1'y = d_i}, link
flows v = A'y, latent gradient g = A t(v). Only the update operator
changes: Euclidean projection onto each OD demand simplex, with a
safeguarded Barzilai--Borwein trial step and Armijo backtracking along
the projected direction.

Deliberately NOT named reduced gradient: no basis, no elimination, no
reduced costs. The stopping certificate is the same Frank--Wolfe duality
gap (computable from the identical LMO), so FW and LPG are stopped and
compared at matched accuracy.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from fw_origin_compression import (
    AtomRepresentation, FWSolveResult, beckmann_from_link_flow, link_cost,
    _lmo, _summarize,
)


def project_od_simplex(y: np.ndarray, demand: float) -> np.ndarray:
    """Euclidean projection onto {z >= 0, 1'z = demand} (sorting method)."""
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - demand
    j = np.arange(1, len(y) + 1)
    rho = np.nonzero(u - css / j > 0)[0]
    if len(rho) == 0:
        return np.full_like(y, demand / len(y))
    theta = css[rho[-1]] / (rho[-1] + 1)
    return np.maximum(y - theta, 0.0)


def latent_projected_gradient(
    problem,
    rep: AtomRepresentation,
    max_iter: int = 20000,
    relative_gap_tol: float = 1e-8,
    y_start: np.ndarray | None = None,
    f_target: float | None = None,
    armijo_sigma: float = 1e-4,
    armijo_beta: float = 0.5,
) -> FWSolveResult:
    y = rep.y0.copy() if y_start is None else y_start.copy()
    v = rep.A.T @ y
    g = np.empty(len(y))
    y_prev = None
    g_prev = None
    eta = 1e-2
    rel_gap = np.inf
    path_cost_evals = 0
    line_search_evals = 0
    time_to_target = None
    start = perf_counter()

    for it in range(1, max_iter + 1):
        t = link_cost(problem, v)
        g[:] = rep.A @ t
        path_cost_evals += len(g)

        # Matched-accuracy certificate: identical FW duality gap.
        s = _lmo(problem, rep, g)
        gap = float(g @ (y - s))
        total_cost = max(float(t @ v), 1e-15)
        rel_gap = max(gap, 0.0) / total_cost
        F = beckmann_from_link_flow(problem, v)
        if f_target is not None and time_to_target is None \
                and F <= f_target:
            time_to_target = perf_counter() - start
            break                      # timed run: target reached
        if rel_gap <= relative_gap_tol:
            break

        # Safeguarded BB1 trial step.
        if y_prev is not None:
            sd = y - y_prev
            yd = g - g_prev
            sy = float(sd @ yd)
            if sy > 1e-16:
                eta = min(max(float(sd @ sd) / sy, 1e-8), 1e4)
        y_prev = y.copy()
        g_prev = g.copy()

        trial = y - eta * g
        proj = trial.copy()
        for i, sl in enumerate(rep.od_atom_slices):
            proj[sl] = project_od_simplex(trial[sl],
                                          float(problem.demand[i]))
        d = proj - y
        dgd = float(g @ d)
        if dgd >= -1e-16:
            break                      # stationary to tolerance
        dv = rep.A.T @ d
        alpha = 1.0
        F0 = F
        while alpha > 1e-12:
            line_search_evals += 1
            if beckmann_from_link_flow(problem, v + alpha * dv) \
                    <= F0 + armijo_sigma * alpha * dgd:
                break
            alpha *= armijo_beta
        y = y + alpha * d
        v = v + alpha * dv

    elapsed = perf_counter() - start
    res = _summarize(
        problem, rep, "Latent-PG", y, v, rel_gap, it, elapsed,
        path_cost_evals, line_search_evals,
    )
    res.__dict__["time_to_target"] = time_to_target
    return res


def frank_wolfe_timed(
    problem,
    rep: AtomRepresentation,
    max_iter: int = 20000,
    relative_gap_tol: float = 1e-8,
    f_target: float | None = None,
) -> FWSolveResult:
    """frank_wolfe with time-to-target tracking; loop semantics identical
    to fw_origin_compression.frank_wolfe."""
    from fw_origin_compression import _exact_bpr_line_search
    y = rep.y0.copy()
    v = rep.A.T @ y
    path_cost_evals = 0
    line_search_evals = 0
    rel_gap = np.inf
    time_to_target = None
    start = perf_counter()
    for it in range(1, max_iter + 1):
        t = link_cost(problem, v)
        atom_cost = rep.A @ t
        path_cost_evals += len(atom_cost)
        s = _lmo(problem, rep, atom_cost)
        d = s - y
        gap = float(atom_cost @ (y - s))
        total_cost = max(float(t @ v), 1e-15)
        rel_gap = max(gap, 0.0) / total_cost
        if f_target is not None and time_to_target is None \
                and beckmann_from_link_flow(problem, v) <= f_target:
            time_to_target = perf_counter() - start
            break                      # timed run: target reached
        if rel_gap <= relative_gap_tol:
            break
        dv = rep.A.T @ d
        step, ne = _exact_bpr_line_search(problem, v, dv)
        line_search_evals += ne
        y += step * d
        v += step * dv
    elapsed = perf_counter() - start
    res = _summarize(
        problem, rep, "Frank-Wolfe", y, v, rel_gap, it, elapsed,
        path_cost_evals, line_search_evals,
    )
    res.__dict__["time_to_target"] = time_to_target
    return res
