#!/usr/bin/env python3
"""Does the affine offset w^0 help?  Controlled ablation (2026-07-19).

The signed-SVD representation is affine: x_N = w^0 + U_r z with w^0 the nominal minor
flow, and feasibility w^0 + U_r z >= 0.  This script toggles ONLY w^0 (affine vs linear
decoder), holding the basis, solver, constraints, rank, and stopping rule fixed:

    AFFINE  u_minor = u0_minor + Q z      (current published representation)
    LINEAR  u_minor =            Q z      (w^0 = 0; z = 0 reproduces the zero vector)

Two competing predictions are tested:
  * near-nominal regimes should favour AFFINE (z = 0 already reproduces the nominal state);
  * anchor-supported equilibria (minor mass exactly zero) should favour LINEAR, because
    the affine offset can only be cancelled by solving U_r z = -w^0 inside a rank-r span.

Reference for the gap is R0-full in every cell, exactly as in the frozen grid.

    python run_w0_ablation.py --mode synthetic      # frozen 32-case grid
    python run_w0_ablation.py --mode real           # network-grounded subproblems
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "real"))

import prototype_static_path_assignment_step2 as core  # noqa: E402


def _solve(problem, structure, decoder):
    """Frozen-grid solver path (SLSQP, analytic gradient, same tolerances)."""
    x, opt, secs = core.solve_decoder(problem, structure, decoder)
    return x, opt, secs


def linear_variant(dec):
    """Same basis and bounds, offset removed (w^0 = 0), latent start at z = 0."""
    theta0 = dec.theta0.copy()
    theta0[dec.n_explicit:] = 0.0
    return replace(dec, name=dec.name + "-nooffset",
                   offset=np.zeros_like(dec.offset), theta0=theta0)


def one_case(problem, x0, rank, major_per_od, tag_fields):
    structure = core.build_reduced_structure(problem, x0)
    u0 = core.nominal_reduced_flow(problem, structure, x0)

    full = core.make_full_decoder(u0)
    xref, _, _ = _solve(problem, structure, full)
    f_ref = core.beckmann_objective(problem, xref)
    v_ref = problem.B.T @ xref
    nref = max(float(np.linalg.norm(v_ref)), 1e-15)

    rows = []
    for weighted in (False, True):
        aff = core.make_svd_decoder(structure, u0, rank, major_per_od, weighted=weighted)
        for variant, dec in (("affine_w0", aff), ("linear_no_w0", linear_variant(aff))):
            x, opt, secs = _solve(problem, structure, dec)
            f = core.beckmann_objective(problem, x)
            v = problem.B.T @ x
            dres = max(abs(float(np.sum(x[sl]) - problem.demand[i]))
                       for i, sl in enumerate(problem.od_slices))
            rows.append(dict(
                **tag_fields, rank=rank,
                basis="W" if weighted else "U", variant=variant,
                n_variables=dec.Q.shape[1],
                objective_gap_pct=100.0 * (f - f_ref) / max(abs(f_ref), 1e-15),
                link_rel_l2=float(np.linalg.norm(v - v_ref) / nref),
                demand_residual=dres, min_path_flow=float(np.min(x)),
                solve_seconds=secs, success=bool(opt.success)))
    # Diagnostics the prediction hinges on: how much mass does the reference optimum put
    # on the MINOR block (the part the SVD must represent), and on non-anchor paths at all?
    major_idx, minor_idx = core.choose_major_minor(structure, u0, major_per_od)
    u_ref = xref[structure.u_global_paths]
    tol = 1e-9 * float(np.max(problem.demand))
    supp = int(np.sum(u_ref > tol))
    minor_mass = float(np.sum(u_ref[minor_idx])) if len(minor_idx) else 0.0
    minor_supp = int(np.sum(u_ref[minor_idx] > tol)) if len(minor_idx) else 0
    total_d = float(np.sum(problem.demand))
    for r in rows:
        r["ref_nonanchor_support"] = supp
        r["ref_minor_support"] = minor_supp
        r["ref_minor_mass_share"] = minor_mass / max(total_d, 1e-15)
    return rows


def synthetic(seed):
    out = []
    for n_od in (1, 2):
        for k in (10, 25):
            for rank in (2, 5):
                for conc in (0.5, 2.5):
                    for pert in (0.0, 0.10):
                        base = core.make_synthetic_problem(n_od, k, seed=seed)
                        x0 = core.make_nominal_x(base, conc)
                        target = core.perturb_demands(base, pert)
                        out.extend(one_case(
                            target, x0, rank, 1,
                            dict(setting="synthetic", network="grid", n_od=n_od,
                                 k_per_od=k, concentration=conc, perturbation=pert)))
    return out


def real(seed, demand_scale=1.0):
    """demand_scale > 1 drives the spread-support regime (4x reproduces the documented
    5-11 non-anchor paths per OD), which is the fairest test of the affine offset."""
    from real_network_loader import make_factory
    out = []
    for net in ("sioux-falls", "chicago-sketch"):
        factory = make_factory(net)
        for k in (8, 16, 32):
            base = factory(3, k, seed=seed)
            x0 = core.make_nominal_x(base, 2.0)
            problem = (base if demand_scale == 1.0
                       else core.perturb_demands(base, demand_scale - 1.0))
            for rank in (2, 5):
                out.extend(one_case(
                    problem, x0, rank, 1,
                    dict(setting="real_x%g" % demand_scale, network=net, n_od=3,
                         k_per_od=k, concentration=2.0,
                         perturbation=demand_scale - 1.0)))
            print("  %s K=%d done" % (net, k), flush=True)
    return out


def summarize(rows):
    hdr = ("%-8s %-13s %-3s %-4s %-3s %-5s %-5s %11s %11s %7s %9s"
           % ("network", "case", "nOD", "K", "r", "conc", "basis",
              "affine w0%", "linear%", "mnrSup", "mnrMass"))
    print("\n" + hdr); print("-" * len(hdr))
    keyf = lambda r: (r["setting"], r["network"], r["n_od"], r["k_per_od"], r["rank"],
                      r["concentration"], r["perturbation"], r["basis"])
    seen = {}
    for r in rows:
        seen.setdefault(keyf(r), {})[r["variant"]] = r
    wins = {"affine": 0, "linear": 0, "tie": 0}
    spread_wins = {"affine": 0, "linear": 0, "tie": 0}
    for k in sorted(seen):
        a = seen[k].get("affine_w0"); l = seen[k].get("linear_no_w0")
        if not (a and l):
            continue
        ga, gl = a["objective_gap_pct"], l["objective_gap_pct"]
        bucket = ("tie" if abs(ga - gl) <= 1e-9 * max(1.0, abs(ga))
                  else ("affine" if ga < gl else "linear"))
        wins[bucket] += 1
        if a["ref_minor_support"] > 0:          # minor block actually carries flow
            spread_wins[bucket] += 1
        print("%-8s %-13s %-3s %-4s %-3s %-5s %-5s %11.4g %11.4g %7d %9.3g"
              % (k[1][:8], "pert=%.2f" % k[6], k[2], k[3], k[4], k[5], k[7],
                 ga, gl, a["ref_minor_support"], a["ref_minor_mass_share"]))
    tot = sum(wins.values())
    print("\nALL cells:      AFFINE better %d/%d | LINEAR better %d/%d | ties %d"
          % (wins["affine"], tot, wins["linear"], tot, wins["tie"]))
    st = sum(spread_wins.values())
    print("Cells where the MINOR block carries flow (the only fair test of w0): %d" % st)
    if st:
        print("                AFFINE better %d/%d | LINEAR better %d/%d | ties %d"
              % (spread_wins["affine"], st, spread_wins["linear"], st,
                 spread_wins["tie"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real", "spread", "both"], default="both")
    ap.add_argument("--demand-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="w0_ablation.csv")
    a = ap.parse_args()
    rows = []
    if a.mode in ("synthetic", "both"):
        print("[synthetic grid]", flush=True)
        rows += synthetic(a.seed)
    if a.mode in ("real", "both"):
        print("[real subproblems]", flush=True)
        rows += real(a.seed)
    if a.mode == "spread":
        print("[real subproblems, demand x%g -> spread support]" % a.demand_scale, flush=True)
        rows += real(a.seed, a.demand_scale)
    summarize(rows)
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote %s (%d rows)" % (a.output, len(rows)))


if __name__ == "__main__":
    main()
