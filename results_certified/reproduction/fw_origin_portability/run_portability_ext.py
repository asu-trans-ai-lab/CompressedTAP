"""Portability extension cells (2026-07-18, this-machine certification).

Three additions to run_solver_portability.py, none of which modify it:

  A. controlled K=2048 (n_od=2, memory limit of the dense prototype):
     the K/m = 256 regime of space-time column pools.
  B. latent mirror descent (entropic geometry): multiplicative per-OD
     updates with a halving step controller, same atoms, same latent
     gradient, same FW duality-gap certificate. Run on controlled K=32
     and K=512; agreement with LPG recorded.
  C. six-way instance check on the exported spread instance
     (interface/instance_cs_k32_spread): C++ grouped BB reference vs
     Python LPG / MD / latent FW on the identical atoms, with LPG
     iteration count recorded.

Outputs: results/solver_portability/portability_ext.csv (+ figure data).
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "suite/src", ROOT / "real", ROOT / "interface", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import prototype_static_path_assignment_step2 as core  # noqa: E402
from fw_origin_compression import (                     # noqa: E402
    beckmann_from_link_flow, link_cost, _lmo,
)
from latent_projected_gradient import (                 # noqa: E402
    latent_projected_gradient, frank_wolfe_timed,
)
from run_solver_portability import run_instance         # noqa: E402
from fw_adapter import load_instance_atoms              # noqa: E402

OUT = ROOT / "fw" / "results" / "solver_portability"
SEED = 7


def latent_mirror_descent(problem, rep, max_iter=20000,
                          relative_gap_tol=1e-8):
    """Entropic mirror descent on the OD demand simplices.

    Multiplicative update s <- s * exp(-eta g), renormalized per OD to
    d_i, with a halving step controller (accept only descent steps).
    Shares atoms, gradient, and the FW duality-gap certificate with
    L+FW / L+PG.
    """
    y = np.maximum(rep.y0.copy(), 1e-300)
    for i, sl in enumerate(rep.od_atom_slices):
        tot = y[sl].sum()
        y[sl] *= problem.demand[i] / tot if tot > 0 else 1.0
        y[sl] = np.maximum(y[sl], 1e-12 * problem.demand[i])
        y[sl] *= problem.demand[i] / y[sl].sum()
    v = rep.A.T @ y
    eta = 1e-2
    rel_gap = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        t = link_cost(problem, v)
        g = rep.A @ t
        s = _lmo(problem, rep, g)
        gap = float(g @ (y - s))
        total_cost = max(float(t @ v), 1e-15)
        rel_gap = max(gap, 0.0) / total_cost
        if rel_gap <= relative_gap_tol:
            break
        F0 = beckmann_from_link_flow(problem, v)
        step = eta
        accepted = False
        for _ in range(60):
            trial = y * np.exp(-step * (g - g.min()))
            for i, sl in enumerate(rep.od_atom_slices):
                tot = trial[sl].sum()
                if tot <= 0:
                    trial[sl] = problem.demand[i] / (sl.stop - sl.start)
                else:
                    trial[sl] *= problem.demand[i] / tot
            v_t = rep.A.T @ trial
            if beckmann_from_link_flow(problem, v_t) < F0:
                y, v = trial, v_t
                eta = min(step * 2.0, 1e4)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return {"objective": beckmann_from_link_flow(problem, v),
            "rel_gap": rel_gap, "iterations": it, "y": y}


def main():
    rows = []

    # --- A: K = 2048, n_od = 2 -------------------------------------
    print("[A] controlled K=2048 (n_od=2)", flush=True)
    problem = core.make_synthetic_problem(2, 2048, seed=SEED)
    run_instance("controlled_K2048_nod2", problem, rows)

    # --- B: mirror descent agreement -------------------------------
    for k in (32, 512):
        problem = core.make_synthetic_problem(4, k, seed=SEED)
        x0 = core.make_nominal_x(problem, 2.0)
        from run_solver_portability import representations
        reps = dict(representations(problem, x0))
        rep = reps["RC"]
        lpg = latent_projected_gradient(problem, rep, max_iter=40000,
                                        relative_gap_tol=1e-10)
        md = latent_mirror_descent(problem, rep, max_iter=40000,
                                   relative_gap_tol=1e-10)
        rel = abs(md["objective"] - lpg.objective) / abs(lpg.objective)
        print(f"[B] K={k}: LPG {lpg.objective:.12e} vs MD "
              f"{md['objective']:.12e} rel {rel:.2e} "
              f"(MD gap {md['rel_gap']:.1e}, {md['iterations']} it)",
              flush=True)
        rows.append({"instance": f"controlled_K{k}",
                     "representation": "MD_vs_LPG_RC", "atoms": "",
                     "n_paths": problem.B.shape[0], "solver": "MD",
                     "restricted_optimum": md["objective"],
                     "representation_gap_pct": rel,
                     "time_to_target_s": "",
                     "iterations": md["iterations"],
                     "demand_residual": "", "min_path_flow": ""})

    # --- C: six-way instance check on the spread instance ----------
    inst = ROOT / "interface" / "instance_cs_k32_spread"
    exe = ROOT / "cpp" / ("tapkernel.exe"
                          if sys.platform.startswith("win")
                          else "tapkernel")
    out = subprocess.run([str(exe), "solve", str(inst)],
                         capture_output=True, text=True,
                         check=True).stdout
    ref = float(dict(l.split() for l in out.strip().splitlines())
                ["grouped_objective"])
    problem, full_rep, grouped_rep = load_instance_atoms(inst)
    lpg = latent_projected_gradient(problem, grouped_rep,
                                    max_iter=40000,
                                    relative_gap_tol=1e-10)
    md = latent_mirror_descent(problem, grouped_rep, max_iter=40000,
                               relative_gap_tol=1e-8)
    fw = frank_wolfe_timed(problem, grouped_rep, max_iter=40000,
                           relative_gap_tol=1e-8)
    for name, obj, its in (("LPG", lpg.objective, lpg.iterations),
                           ("MD", md["objective"], md["iterations"]),
                           ("FW", fw.objective, fw.iterations)):
        rel = abs(obj - ref) / abs(ref)
        print(f"[C] spread atoms {name}: obj {obj:.12e} rel-to-cppBB "
              f"{rel:.2e} ({its} it)", flush=True)
        rows.append({"instance": "cs_k32_spread",
                     "representation": f"{name}_vs_cppBB", "atoms": "",
                     "n_paths": problem.B.shape[0], "solver": name,
                     "restricted_optimum": obj,
                     "representation_gap_pct": rel,
                     "time_to_target_s": "", "iterations": its,
                     "demand_residual": "", "min_path_flow": ""})

    with (OUT / "portability_ext.csv").open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
