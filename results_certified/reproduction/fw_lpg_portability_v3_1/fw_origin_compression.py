"""Feasible path-space compression for Frank--Wolfe and origin-block assignment.

The key rule is that compression must preserve the traffic-assignment feasible
geometry.  A signed SVD coordinate is not, by itself, a Frank--Wolfe atom.
This module therefore provides two FW-compatible uses of compression:

1. Grouped nonnegative atoms.  Every atom is a convex path-flow distribution
   within one OD pair.  The FW linear minimization oracle (LMO) simply chooses
   the cheapest atom for each OD, so demand conservation and nonnegativity are
   exact at every iterate.
2. Flow-weighted path screening.  Weighted pivoted QR selects a smaller set of
   actual paths.  Standard FW then solves the exact restricted master problem
   over those feasible path columns.

It also provides an origin-block conditional-gradient prototype.  This is a
transparent origin-decomposed analogue, not a claim to reproduce every detail
of a production Bar-Gera bush/origin-based implementation.  It maintains
origin-specific link-flow blocks and updates one origin at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import qr

import prototype_static_path_assignment_step2 as core


@dataclass
class AtomRepresentation:
    name: str
    # Online atom-link signatures: one row per feasible atom.
    A: np.ndarray
    # Optional path decoder x = P y.  Full and selected-path reps can decode
    # without a dense matrix using path_indices.
    P: np.ndarray | None
    path_indices: np.ndarray | None
    od_atom_slices: List[slice]
    y0: np.ndarray
    preprocess_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class FWSolveResult:
    method: str
    representation: str
    objective: float
    final_relative_gap: float
    iterations: int
    solve_seconds: float
    n_atoms: int
    n_links: int
    online_bytes: int
    demand_residual: float
    min_path_flow: float
    x: np.ndarray
    y: np.ndarray
    link_flow: np.ndarray
    path_cost_evaluations: int
    line_search_evaluations: int


def beckmann_from_link_flow(problem: core.Problem, v: np.ndarray) -> float:
    ratio = np.maximum(v / problem.capacity, 0.0)
    return float(np.sum(
        problem.t0 * (
            v
            + problem.alpha * problem.capacity
            * ratio ** (problem.beta + 1.0) / (problem.beta + 1.0)
        )
    ))


def link_cost(problem: core.Problem, v: np.ndarray) -> np.ndarray:
    ratio = np.maximum(v / problem.capacity, 0.0)
    return problem.t0 * (1.0 + problem.alpha * ratio ** problem.beta)


def _decode_path_flow(
    problem: core.Problem,
    rep: AtomRepresentation,
    y: np.ndarray,
) -> np.ndarray:
    if rep.P is not None:
        return rep.P @ y
    if rep.path_indices is None:
        # Full path representation: one atom per path in original order.
        return y.copy()
    x = np.zeros(problem.B.shape[0])
    x[rep.path_indices] = y
    return x


def _online_bytes(rep: AtomRepresentation) -> int:
    total = int(rep.A.nbytes + rep.y0.nbytes)
    # Group atoms require their path decoder only for reporting/reconstruction;
    # the online FW kernel uses A directly.  Do not count P as online state.
    if rep.path_indices is not None:
        total += int(rep.path_indices.nbytes)
    return total


def _validate_atoms(problem: core.Problem, rep: AtomRepresentation) -> None:
    if rep.A.shape[0] != len(rep.y0):
        raise ValueError("atom signature and coordinate size mismatch")
    if len(rep.od_atom_slices) != len(problem.od_slices):
        raise ValueError("one atom block is required per OD")
    for i, sl in enumerate(rep.od_atom_slices):
        if sl.stop <= sl.start:
            raise ValueError(f"OD {i} has no feasible atom")
        if abs(float(np.sum(rep.y0[sl])) - problem.demand[i]) > 1e-8:
            raise ValueError(f"OD {i} initial atom masses are not feasible")
    if rep.P is not None:
        col_sums = np.sum(rep.P, axis=0)
        if np.max(np.abs(col_sums - 1.0)) > 1e-10:
            raise ValueError("every atom must be a path-flow distribution summing to one")
        if np.min(rep.P) < -1e-12:
            raise ValueError("FW-compatible atoms must be nonnegative")
        if np.max(np.abs(rep.P.T @ problem.B - rep.A)) > 1e-9:
            raise ValueError("atom-link signatures do not match the path decoder")


def make_full_path_representation(
    problem: core.Problem,
    x0: np.ndarray | None = None,
) -> AtomRepresentation:
    start = perf_counter()
    if x0 is None:
        x0 = core.make_nominal_x(problem, 2.0)
    rep = AtomRepresentation(
        name="FW-full-path-pool",
        A=problem.B.copy(),
        P=None,
        path_indices=None,
        od_atom_slices=list(problem.od_slices),
        y0=x0.copy(),
        preprocess_seconds=perf_counter() - start,
        metadata={"type": "full", "n_paths": problem.B.shape[0]},
    )
    _validate_atoms(problem, rep)
    return rep


def make_grouped_atom_representation(
    problem: core.Problem,
    x0: np.ndarray | None = None,
    groups_per_od: int = 4,
    major_per_od: int = 1,
) -> AtomRepresentation:
    """Build nonnegative OD-local atoms with precomputed link signatures."""
    start = perf_counter()
    if x0 is None:
        x0 = core.make_nominal_x(problem, 2.0)

    columns: List[np.ndarray] = []
    y0_parts: List[float] = []
    od_atom_slices: List[slice] = []
    cursor = 0

    for sl in problem.od_slices:
        paths = np.arange(sl.start, sl.stop)
        order = paths[np.argsort(-x0[paths])]
        n_major = min(max(major_per_od, 0), len(paths))
        major = order[:n_major]
        minor = order[n_major:]

        for p in major:
            col = np.zeros(problem.B.shape[0])
            col[p] = 1.0
            columns.append(col)
            y0_parts.append(float(x0[p]))

        if len(minor):
            labels = core.deterministic_kmeans(
                problem.B[minor], min(groups_per_od, len(minor))
            )
            for label in np.unique(labels):
                members = minor[labels == label]
                nominal = x0[members]
                total = float(np.sum(nominal))
                shares = (
                    nominal / total
                    if total > 1e-14
                    else np.full(len(members), 1.0 / len(members))
                )
                col = np.zeros(problem.B.shape[0])
                col[members] = shares
                columns.append(col)
                y0_parts.append(total)

        # Degenerate protection: if no major and no group, keep one path.
        if cursor == len(columns):
            p = int(paths[0])
            col = np.zeros(problem.B.shape[0])
            col[p] = 1.0
            columns.append(col)
            y0_parts.append(float(problem.demand[len(od_atom_slices)]))

        od_atom_slices.append(slice(cursor, len(columns)))
        cursor = len(columns)

    P = np.column_stack(columns)
    y0 = np.asarray(y0_parts, dtype=float)
    # Numerical normalization of each OD block.
    for i, sl in enumerate(od_atom_slices):
        total = float(np.sum(y0[sl]))
        if total <= 0:
            y0[sl] = problem.demand[i] / (sl.stop - sl.start)
        else:
            y0[sl] *= problem.demand[i] / total

    rep = AtomRepresentation(
        name="FW-grouped-atoms",
        A=P.T @ problem.B,
        P=P,
        path_indices=None,
        od_atom_slices=od_atom_slices,
        y0=y0,
        preprocess_seconds=perf_counter() - start,
        metadata={
            "type": "grouped",
            "groups_per_od": int(groups_per_od),
            "major_per_od": int(major_per_od),
            "n_atoms": int(P.shape[1]),
        },
    )
    _validate_atoms(problem, rep)
    return rep


def make_qr_screened_representation(
    problem: core.Problem,
    x0: np.ndarray | None = None,
    paths_per_od: int = 8,
    weighted: bool = True,
) -> AtomRepresentation:
    """Select actual feasible paths using pivoted QR of path incidence rows.

    This is the safe way to use a spectral/low-rank diagnostic with classic
    Frank--Wolfe: the selected columns remain actual paths, so the standard
    shortest-column LMO and simplex feasibility are unchanged.
    """
    start = perf_counter()
    if x0 is None:
        x0 = core.make_nominal_x(problem, 2.0)

    selected: List[int] = []
    y0_parts: List[float] = []
    od_atom_slices: List[slice] = []
    cursor = 0

    for i, sl in enumerate(problem.od_slices):
        paths = np.arange(sl.start, sl.stop)
        m = min(max(paths_per_od, 1), len(paths))
        X = problem.B[paths].astype(float)
        if weighted:
            positive = x0[paths][x0[paths] > 0]
            scale = float(np.mean(positive)) if len(positive) else 1.0
            floor = max(1e-4 * scale, 1e-12)
            weights = np.sqrt(np.maximum(x0[paths], floor))
            X = weights[:, None] * X

        # Pivot columns of X.T, which correspond to rows/paths of X.
        _, _, piv = qr(X.T, pivoting=True, mode="economic")
        chosen_local = list(map(int, piv[:m]))
        # Always include the nominal major path; replace last pivot if needed.
        major_local = int(np.argmax(x0[paths]))
        if major_local not in chosen_local:
            if len(chosen_local) == m:
                chosen_local[-1] = major_local
            else:
                chosen_local.append(major_local)
        chosen_local = list(dict.fromkeys(chosen_local))
        if len(chosen_local) < m:
            for idx in np.argsort(-x0[paths]):
                if int(idx) not in chosen_local:
                    chosen_local.append(int(idx))
                if len(chosen_local) >= m:
                    break

        chosen = paths[np.asarray(chosen_local[:m], dtype=int)]
        selected.extend(chosen.tolist())
        nominal = x0[chosen].copy()
        total = float(np.sum(nominal))
        if total <= 0:
            nominal[:] = problem.demand[i] / len(chosen)
        else:
            nominal *= problem.demand[i] / total
        y0_parts.extend(nominal.tolist())
        od_atom_slices.append(slice(cursor, cursor + len(chosen)))
        cursor += len(chosen)

    selected_arr = np.asarray(selected, dtype=int)
    rep = AtomRepresentation(
        name=(
            "FW-weighted-QR-screened"
            if weighted else "FW-unweighted-QR-screened"
        ),
        A=problem.B[selected_arr].copy(),
        P=None,
        path_indices=selected_arr,
        od_atom_slices=od_atom_slices,
        y0=np.asarray(y0_parts, dtype=float),
        preprocess_seconds=perf_counter() - start,
        metadata={
            "type": "selected_paths",
            "weighted": bool(weighted),
            "paths_per_od": int(paths_per_od),
        },
    )
    _validate_atoms(problem, rep)
    return rep


def _lmo(
    problem: core.Problem,
    rep: AtomRepresentation,
    atom_cost: np.ndarray,
    od_subset: Sequence[int] | None = None,
) -> np.ndarray:
    s = np.zeros_like(atom_cost)
    indices = range(len(problem.od_slices)) if od_subset is None else od_subset
    for i in indices:
        sl = rep.od_atom_slices[i]
        j = sl.start + int(np.argmin(atom_cost[sl]))
        s[j] = problem.demand[i]
    return s


def _exact_bpr_line_search(
    problem: core.Problem,
    v: np.ndarray,
    dv: np.ndarray,
    max_iter: int = 60,
) -> Tuple[float, int]:
    """Exact one-dimensional search via monotone derivative bisection."""
    if np.linalg.norm(dv) <= 1e-16:
        return 0.0, 0

    def derivative(lam: float) -> float:
        vv = np.maximum(v + lam * dv, 0.0)
        return float(link_cost(problem, vv) @ dv)

    evals = 1
    g0 = derivative(0.0)
    if g0 >= 0.0:
        return 0.0, evals
    g1 = derivative(1.0)
    evals += 1
    if g1 <= 0.0:
        return 1.0, evals

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gm = derivative(mid)
        evals += 1
        if gm <= 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12:
            break
    return 0.5 * (lo + hi), evals


def _summarize(
    problem: core.Problem,
    rep: AtomRepresentation,
    method: str,
    y: np.ndarray,
    v: np.ndarray,
    rel_gap: float,
    iterations: int,
    elapsed: float,
    path_cost_evals: int,
    line_search_evals: int,
) -> FWSolveResult:
    x = _decode_path_flow(problem, rep, y)
    residual = max(
        abs(float(np.sum(x[sl])) - problem.demand[i])
        for i, sl in enumerate(problem.od_slices)
    )
    return FWSolveResult(
        method=method,
        representation=rep.name,
        objective=beckmann_from_link_flow(problem, v),
        final_relative_gap=float(rel_gap),
        iterations=int(iterations),
        solve_seconds=float(elapsed),
        n_atoms=len(y),
        n_links=problem.B.shape[1],
        online_bytes=_online_bytes(rep),
        demand_residual=float(residual),
        min_path_flow=float(np.min(x)),
        x=x,
        y=y.copy(),
        link_flow=v.copy(),
        path_cost_evaluations=int(path_cost_evals),
        line_search_evaluations=int(line_search_evals),
    )


def frank_wolfe(
    problem: core.Problem,
    rep: AtomRepresentation,
    max_iter: int = 5000,
    relative_gap_tol: float = 1e-8,
    y_start: np.ndarray | None = None,
) -> FWSolveResult:
    """Classic Frank--Wolfe on a full or compressed feasible atom simplex."""
    y = rep.y0.copy() if y_start is None else y_start.copy()
    v = rep.A.T @ y
    path_cost_evals = 0
    line_search_evals = 0
    rel_gap = np.inf
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
        if rel_gap <= relative_gap_tol:
            break
        dv = rep.A.T @ d
        step, ne = _exact_bpr_line_search(problem, v, dv)
        line_search_evals += ne
        y += step * d
        v += step * dv

    elapsed = perf_counter() - start
    return _summarize(
        problem, rep, "Frank-Wolfe", y, v, rel_gap, it, elapsed,
        path_cost_evals, line_search_evals,
    )


def origin_block_frank_wolfe(
    problem: core.Problem,
    rep: AtomRepresentation,
    od_origin_group: Sequence[int],
    max_sweeps: int = 2000,
    relative_gap_tol: float = 1e-8,
) -> FWSolveResult:
    """Cyclic origin-block conditional gradient.

    Each origin owns several OD simplices.  The method stores an origin-specific
    link-flow block and updates one origin at a time with exact line search.
    This captures the conservation and decomposition interface needed by
    origin/bush methods while remaining compact enough for a reference test.
    """
    od_origin_group = np.asarray(od_origin_group, dtype=int)
    if len(od_origin_group) != len(problem.od_slices):
        raise ValueError("od_origin_group must contain one entry per OD")
    origins = np.unique(od_origin_group)
    od_by_origin = {
        int(o): np.flatnonzero(od_origin_group == o).tolist() for o in origins
    }

    y = rep.y0.copy()
    origin_atom_indices: Dict[int, np.ndarray] = {}
    v_by_origin: Dict[int, np.ndarray] = {}
    for o in origins:
        atom_idx: List[int] = []
        for i in od_by_origin[int(o)]:
            sl = rep.od_atom_slices[i]
            atom_idx.extend(range(sl.start, sl.stop))
        idx = np.asarray(atom_idx, dtype=int)
        origin_atom_indices[int(o)] = idx
        v_by_origin[int(o)] = rep.A[idx].T @ y[idx]
    v = np.sum(np.vstack(list(v_by_origin.values())), axis=0)

    path_cost_evals = 0
    line_search_evals = 0
    rel_gap = np.inf
    start = perf_counter()

    for sweep in range(1, max_sweeps + 1):
        for o in origins:
            ods = od_by_origin[int(o)]
            idx = origin_atom_indices[int(o)]
            t = link_cost(problem, v)
            atom_cost = rep.A @ t
            path_cost_evals += len(idx)
            s_full = _lmo(problem, rep, atom_cost, od_subset=ods)
            d_local = s_full[idx] - y[idx]
            if np.linalg.norm(d_local) <= 1e-16:
                continue
            dv = rep.A[idx].T @ d_local
            step, ne = _exact_bpr_line_search(problem, v, dv)
            line_search_evals += ne
            y[idx] += step * d_local
            v_by_origin[int(o)] += step * dv
            v += step * dv

        # Global FW gap after one origin sweep.
        t = link_cost(problem, v)
        atom_cost = rep.A @ t
        s = _lmo(problem, rep, atom_cost)
        gap = float(atom_cost @ (y - s))
        rel_gap = max(gap, 0.0) / max(float(t @ v), 1e-15)
        path_cost_evals += len(atom_cost)
        if rel_gap <= relative_gap_tol:
            break

    elapsed = perf_counter() - start
    return _summarize(
        problem, rep, "Origin-block FW", y, v, rel_gap, sweep, elapsed,
        path_cost_evals, line_search_evals,
    )
