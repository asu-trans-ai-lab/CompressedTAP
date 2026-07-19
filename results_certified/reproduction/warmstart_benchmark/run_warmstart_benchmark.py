"""Warm-start benchmark + scenario break-even (one-table experiment).

Setting (the repeated-scenario use case, measured end to end):
  base scenario  : network-extracted instance (largest OD pool the dense
                   prototype holds), solved cold from the synthetic
                   nominal loading -> this solve IS the nominal
                   equilibrium; its time is T_nominal.
  preprocessing  : T_pool  = path-pool generation (factory build + K
                   shortest-path pools);
                   T_atoms = grouped-atom construction (RC, m=8,
                   shares from the SOLVED base flows).
  scenario stream: 5 demand perturbations (+/-5%, per-OD uniform,
                   seeds 101..105). Per scenario, three timed cells,
                   all with the SAME solver (LPG) and the
                   matched-accuracy rule of the portability protocol
                   (own restricted optimum * (1 + 1e-6)):
    E-cold : full explicit pool, start from the synthetic nominal;
    E-warm : full explicit pool, start from the PREVIOUS scenario's
             solution, per-OD rescaled to the new demands;
    RC-warm: grouped atoms, start from the previous RC solution,
             per-OD rescaled.
  break-even     : N_break = (T_nominal + T_pool + T_atoms)
                             / (mean T_E-warm - mean T_RC-warm),
                   infinity if the denominator is <= 0 (i.e., the
                   warm-started full solve is already faster, and
                   compression never repays its preprocessing).
  The RC representation gap (vs the E optimum, same scenario) is
  reported separately and never folded into a speedup.

Output: results/warmstart/warmstart_benchmark.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "suite/src", ROOT / "real", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import prototype_static_path_assignment_step2 as core  # noqa: E402
from real_network_loader import make_factory            # noqa: E402
from fw_origin_compression import (                     # noqa: E402
    make_full_path_representation, make_grouped_atom_representation,
)
from latent_projected_gradient import (                 # noqa: E402
    latent_projected_gradient,
)

OUT = ROOT / "fw" / "results" / "warmstart"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 7
M_PER_OD = 8
N_SCEN = 5
PERT = 0.05

CASES = [("sioux-falls", 199, 32), ("chicago-sketch", 400, 32)]


def with_demand(problem, demand):
    return core.Problem(B=problem.B, od_slices=problem.od_slices,
                        demand=demand, t0=problem.t0,
                        capacity=problem.capacity,
                        alpha=problem.alpha, beta=problem.beta)


def rescale_per_od(rep, y_prev, problem_new):
    y = y_prev.copy()
    for i, sl in enumerate(rep.od_atom_slices):
        tot = y[sl].sum()
        if tot > 0:
            y[sl] *= float(problem_new.demand[i]) / tot
        else:
            y[sl] = float(problem_new.demand[i]) / (sl.stop - sl.start)
    return y


def solve(problem, rep, y_start=None, tol=1e-9, f_target=None):
    t = perf_counter()
    r = latent_projected_gradient(problem, rep, max_iter=60000,
                                  relative_gap_tol=tol,
                                  y_start=y_start, f_target=f_target)
    return r, perf_counter() - t


def main():
    rows = []
    for net, n_od, k in CASES:
        print(f"[{net}] n_od={n_od} K={k}", flush=True)
        t = perf_counter()
        factory = make_factory(net)
        factory.target_vc = 2.0
        problem = factory(n_od, k, seed=SEED)
        t_pool = perf_counter() - t

        e_rep = make_full_path_representation(
            problem, core.make_nominal_x(problem, 2.0))

        # nominal equilibrium = cold solve of the base scenario
        r_nom, t_nominal = solve(problem, e_rep)
        x_nom = r_nom.y  # E rep: atom masses ARE path flows

        t = perf_counter()
        rc_rep = make_grouped_atom_representation(
            problem, x_nom, groups_per_od=M_PER_OD - 1, major_per_od=1)
        t_atoms = perf_counter() - t

        y_e_prev = x_nom.copy()
        y_rc_prev = rc_rep.y0.copy()
        te_cold, te_warm, trc_warm, gaps = [], [], [], []
        for s in range(1, N_SCEN + 1):
            rng = np.random.default_rng(100 + s)
            d_s = problem.demand * (
                1.0 + PERT * rng.uniform(-1.0, 1.0, len(problem.demand)))
            prob_s = with_demand(problem, d_s)

            # matched-accuracy targets per representation
            fE, _ = solve(prob_s, e_rep,
                          y_start=rescale_per_od(e_rep, x_nom, prob_s))
            fRC, _ = solve(prob_s, rc_rep,
                           y_start=rescale_per_od(rc_rep, rc_rep.y0,
                                                  prob_s))
            tgtE = fE.objective * (1 + 1e-6)
            tgtRC = fRC.objective * (1 + 1e-6)

            rE_cold, tEc = solve(
                prob_s, e_rep,
                y_start=rescale_per_od(
                    e_rep, core.make_nominal_x(prob_s, 2.0), prob_s),
                tol=1e-12, f_target=tgtE)
            rE_warm, tEw = solve(
                prob_s, e_rep,
                y_start=rescale_per_od(e_rep, y_e_prev, prob_s),
                tol=1e-12, f_target=tgtE)
            rRC_warm, tRCw = solve(
                prob_s, rc_rep,
                y_start=rescale_per_od(rc_rep, y_rc_prev, prob_s),
                tol=1e-12, f_target=tgtRC)
            y_e_prev = rE_warm.y
            y_rc_prev = rRC_warm.y

            gap = 100 * (fRC.objective - fE.objective) / abs(fE.objective)
            te_cold.append(tEc)
            te_warm.append(tEw)
            trc_warm.append(tRCw)
            gaps.append(gap)
            print(f"  s{s}: E-cold {tEc:.4f}s E-warm {tEw:.4f}s "
                  f"RC-warm {tRCw:.4f}s gap {gap:.3f}%", flush=True)

        me_c, me_w, mrc = (float(np.mean(te_cold)),
                           float(np.mean(te_warm)),
                           float(np.mean(trc_warm)))
        denom = me_w - mrc
        prep = t_nominal + t_pool + t_atoms
        n_break = prep / denom if denom > 0 else float("inf")
        rows.append({
            "network": net, "n_od": n_od, "k_per_od": k,
            "n_paths": problem.B.shape[0],
            "rc_atoms": rc_rep.A.shape[0],
            "t_pool_s": t_pool, "t_nominal_s": t_nominal,
            "t_atoms_s": t_atoms,
            "mean_E_cold_s": me_c, "mean_E_warm_s": me_w,
            "mean_RC_warm_s": mrc,
            "rc_gap_pct_mean": float(np.mean(gaps)),
            "n_break": n_break,
        })
        print(f"  prep {prep:.3f}s | E-warm {me_w:.4f}s vs RC-warm "
              f"{mrc:.4f}s -> N_break {n_break:.1f}", flush=True)

    with (OUT / "warmstart_benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
