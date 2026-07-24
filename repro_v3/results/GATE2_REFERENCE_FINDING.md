# Gate 2 failure: the tau=0 ALM run is not a valid reference

Sioux Falls SFK25 (6,464 paths / 528 OD), single thread, 2026-07-19.
Certificate = relative Frank-Wolfe duality gap of the feasibility-converted iterate;
this is an OPTIMALITY measure, evaluated identically for every method.

| solution | certificate | objective | vs best |
|---|---|---|---|
| ALM `tol=1e-8`, `maxiter_inner=200` (certified default) | 1.14e-03 | 4,233,374.255 | **+0.0482 %** |
| ALM `tol=1e-8`, `maxiter_inner=2000` | 7.53e-04 | 4,232,323.685 | +0.0234 % |
| Frank-Wolfe `tol=1e-6` (96,758 it, 219.4 s) | 9.99e-07 | 4,231,340 (approx) | +0.0000 % |
| **Gradient projection `tol=1e-9` (186 it, 2.05 s)** | **7.55e-10** | **4,231,335.287** | **0 (best)** |

## What was ruled out

- **Not infeasibility.** Every method converts with `delta_F = 0.0000 %` and demand
  residual 1e-12..1e-13. The v3 conversion is not doing any work here.
- **Not the outer feasibility tolerance.** Tightening ALM `tol` from 1e-6 to 1e-12
  (CPU 2.55 s -> 11.34 s) leaves the certificate pinned at 1.1355e-03 and moves the
  objective only in the 10th significant digit. `tol` is a *feasibility* tolerance
  (`cons = ||Ax-d|| < tol`); it says nothing about optimality.
- **Not the inner budget alone.** `maxiter_inner` 200 -> 2000 improves the certificate
  to 7.53e-04 and the objective by 0.025 %, but 2000 -> 20000 changes nothing. A second
  plateau remains.
- **Not one solver misbehaving.** FW and GP converge independently to the same point
  (Gap_F -0.048048 % vs -0.048164 %, agreeing to 1.2e-4 percentage points) and both
  disagree with ALM in the same direction.

## Consequence for the manuscript

v3 states that the revised experiments "use a consistently converged uncompressed
formulation as the primary reference", and Panel A takes the tau=0 **ALM** run as that
reference. That reference sits ~0.048 % above the optimum, so:

1. any compressed run landing nearer the true optimum reports a **negative** `Gap_F` --
   precisely the artifact the v3 revision exists to remove;
2. every `Gap_F` in Panel A, Panel B and Table 5 is biased low by roughly that amount;
3. the claim that `Gap_F` is "nonnegative up to the common solver tolerance" cannot hold.

## Recommendation (author decision required)

Use **full-path gradient projection converged to a tight duality gap** as `v^ref`, rather
than the tau=0 ALM run. It is still the uncompressed formulation, so the manuscript's
wording is satisfied; it is six orders of magnitude better converged; and it is *cheaper*
(2.05 s vs 5.21 s on this instance). ALM then appears in Table 2 as one of the compared
operators, evaluated against that reference like the others -- which is what the table is
for.

Producing scripts: `python/run_table2_consistency.py`, results
`results/table2_consistency_sioux.csv`.
