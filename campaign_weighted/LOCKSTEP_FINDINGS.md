# The two limits of global-SVD compression: reduction ceiling and representation floor

*Lockstep certification note, 2026-07-27. All numbers Python (`compressed_assignment.py`,
scipy L-BFGS-B inner solver), single process, single thread, Chicago Sketch K10 pool
(n = 903,516 paths, 81,435 OD pairs, 2,950 links), scenario d05 (demand x (1 + U(-0.05, 0.05)),
seed 100, the same instance as the C++ exp07 traces), tau = 4.54, weighted basis, w0 anchored
at the common start. Backing data: `results/exp08_lockstep_K10_d05_r6.csv`,
`results/exp09_controls_K10_d05.csv`. Drivers: `drivers/exp08_lockstep.py`,
`drivers/exp09_controls.py`.*

## 1. The two limits

**Size ceiling (established by exp06, Regional tau sweep):**

    R_max ~ (n - ell - r)/n ~ 1 - 1/Kbar

At least one explicit path must remain per OD pair, so route-pool richness caps the
achievable dimensional reduction before any tuning. Regional-E0 (Kbar = 7.48) saturates at
86.6% measured against an 86.6% ceiling.

**Accuracy floor (established here):**

    G_min(U, r) ~ 0.013     for the tested global-SVD representation, r in {6, 10, 20}

Even when the reduction is sufficient for cheaper iterations, the global signed low-rank
decoder imposes an equilibrium-accuracy floor. Under identical initialization, matched ALM
operators, and an independent full-space evaluator, the rank-6 compressed trajectory
plateaus at common gap G ~ 1.3e-2 while full-space ALM continues below 1e-3.

The two limits are independent: one is a statement about *size*, the other about
*accuracy*. A pool can pass the ceiling and still be bound by the floor (K10 does).

## 2. The lockstep instrument (exp08)

Both solvers advance ONE ALM outer iteration at a time under a single controller, from a
byte-identical iteration 0, and are scored after every synchronized step by a third,
independently implemented full-space evaluator (it calls neither solver's gap code):

- identical start: x_start = project_feasible(max(x0, 1e-3)); compressed gets
  w0 = x_start[minor], z0 = 0, y0 = x_start[major], with d_eff / v_base rebuilt from this
  w0. Identity gate: max|x_F^0 - x_C^0|, link-flow, OD and objective differences all
  exactly 0.0 (asserted, not assumed);
- same demand, path pool, BPR, tolerance (1e-4), inner solver (L-BFGS-B, maxiter 200,
  gtol 1e-5), rho0 = 100, lam0 = 0; the stepping code is copied verbatim from the
  certified `solve_full` / `solve_compressed(hard)`, preserving each side's own
  break/update ordering;
- evaluator metrics on the raw decoded path-flow vector: Beckmann Z; common gap
  G = sum_q sum_p x_p (c_p - c_q^min) / sum_q d_q c_q^min with pool-min costs; OD residual
  (mean and max forms); projected-gradient residual r_PG via per-OD Euclidean simplex
  projection (alpha = 1e-2, fixed and stated); ALM decomposition Z / lam'r / 0.5 rho|r|^2
  / hinge reported separately, never one blended cost;
- per-kernel CPU timers inside each side's objective/gradient callback.

### The ten synchronized steps (K10-d05, r = 6)

| k | G_full | G_comp | r_PG full | r_PG comp | r_OD full | r_OD comp | cpu_f (s) | cpu_c (s) |
|--:|-------:|-------:|----------:|----------:|----------:|----------:|----------:|----------:|
| 1 | 0.00503 | 0.01563 | 0.00278 | 0.00275 | 0.02453 | 0.02104 | 41.5 | 15.6 |
| 2 | 0.00277 | 0.01420 | 0.00069 | 0.00117 | 0.00363 | 0.00320 | 38.8 | 15.8 |
| 3 | 0.00146 | 0.01343 | 0.00047 | 0.00107 | 0.00215 | 0.00189 | 37.1 | 13.9 |
| 4 | 0.00110 | 0.01358 | 0.00032 | 0.00105 | 0.00134 | 0.00118 | 37.5 | 14.2 |
| 5 | 0.00091 | 0.01365 | 0.00021 | 0.00104 | 0.00080 | 0.00074 | 37.7 | 11.1 |
| 6 | 0.00079 | 0.01393 | 0.00014 | 0.00104 | 0.00044 | 0.00045 | 38.6 | 15.8 |
| 7 | 0.00074 | 0.01370 | 0.00009 | 0.00103 | 0.00022 | 0.00035 | 39.6 | 21.3 |
| 8 | 0.00070 | 0.01339 | 0.00005 | 0.00102 | 0.00010 | 0.00037 | 58.0 | 21.4 |
| 9 | 0.00069 | 0.01338 | 0.00002 | 0.00102 | 0.00001 | 0.00033 | 64.0 | 2.2 |
| 10 | 0.00069 | 0.01338 | 0.00003 | 0.00102 | 0.00000 | 0.00032 | 50.8 | 1.3 |

OD feasibility is comparable throughout; the divergence is purely in equilibrium quality.
The persistent full-space PG residual on the compressed side (pinned at ~1.0e-3) shows a
genuine descent direction remains available outside the compressed trajectory at every step.

### Kernel CPU totals over the run

| kernel | full (242.9 s) | compressed (99.2 s) |
|---|---:|---:|
| gradient (major/full) | 145.7 (60%) | 11.4 (12%) |
| link aggregation | 86.6 (36%) | 7.1 (7%) |
| **minor decode + hinge** | -- | **73.4 (74%)** |
| OD residual | 7.7 (3%) | 3.6 (4%) |
| BPR | 2.9 (1%) | 2.1 (2%) |
| latent gradient | -- | 1.6 (2%) |

Three structural answers: (1) the variables were genuinely compressed -- the full side's
dominant kernels shrink from 96% to 19%; (2) but every objective/gradient call still
touches all 811,756 minor rows through the dense O(n_minor x r) decode + hinge, which
becomes 74% of compressed CPU, so the naive 10x dimension ratio delivers only 2.4x
per-outer cost; (3) the OD loop is NOT the bottleneck in this engine (3-4% both sides).

## 3. The five controls (exp09), all branched from the same in-process k = 4 state

After the first outer iteration lam_F != lam_C (different subproblems from k = 2 on), and
the hinge exists only on the compressed side -- so exp08 alone attributes the plateau to
the complete compressed trajectory, not yet to the representation. The controls separate
the three candidates.

| # | control | result | verdict |
|---|---|---|---|
| C1 | shared-dual replay, common (lam, rho), one primal solve, no dual update | lam_F-for-both: G_full 0.00091 / G_comp 0.01319; lam_C-for-both: 0.00092 / 0.01383 | dual drift **eliminated** -- full escapes under either side's multipliers, compressed remains near the plateau under either |
| C2 | hinge accounting + hinge-off step at fixed duals | per-step H: 3.96e4, 5.8e3, 4.2e3, 2.0e3; at k=4: H = 4.71e3, \|\|grad_z H\|\| = 2.28e3, min(u) = -1.08e-2; hinge-off step: G = **2.218** | hinge is **necessary but not the root cause** -- it is part of the mechanism (see below), and removing it admits destructive directions that drive minor flows negative |
| C3 | gradient geometry at the plateau: g_T = per-OD-centered path costs; project onto the compressed tangent | eta_captured = 0.134, eta_missing = 0.866; 87.7% of \|\|g_T\|\|^2 lies in the minor block, of which span(U_6) captures **1.26%** | the representation is the root cause |
| C4 | full-space escape step from the identical compressed iterate and identical ALM state (comp's lam, rho) vs a paired normal compressed step | escape: G 0.01352 -> **0.00277** in one step (84.9 s); paired compressed step: 0.01383 (27.9 s) | **decisive** -- only the representation changed |
| C5 | rank replay from iteration 0, fresh bases, same weights | plateaus: r=6 0.0134, r=10 0.0135, r=20 0.0127 | increasing rank 6 -> 20 produces only a small reduction in the floor; the limitation persists over the tested rank range |

Basis-reuse note: the exp08 basis was built fresh from x_start on the perturbed instance,
so a reused-basis floor is already excluded for this run.

## 4. The mechanism, stated precisely

The compressed feasible set for the minors is the intersection

    F_U = { w0 + U z : w0 + U z >= 0 },

not the linear span alone. The certified statement is:

> The global latent basis poorly aligns with the remaining OD-specific equilibrium
> directions -- at the r = 6 plateau, span(U_6) captures only 1.26% of the remaining
> minor-path equilibrium-correcting gradient -- and the small aligned component is further
> restricted by the nonnegativity boundary of the decoded minor flows (min(u) ~ -1e-2 at
> the plateau; hinge-off G = 2.22 shows what that boundary is holding back).

This is stronger and more accurate than either "rank too small" or "the hinge caused the
problem". Do NOT extrapolate the 1.26% to other ranks: the projection ratio was computed
at r = 6 only; for r = 10 and 20 the evidence is the plateau level, not the ratio.

Conclusion sentence: the compressed solver is computationally cheaper per outer iteration,
but the global signed low-rank representation cannot adequately express the OD-specific,
within-route-set reallocations required for high-accuracy equilibrium, at the tested ranks.

## 5. What this does to the earlier speedup claims

The 3.06x / 3.26x C++ measurements (exp05, Sketch K10/K15) remain valid as runtime
measurements. They must now be qualified:

- both the compressed and the full C++ ALM implementations in those comparisons operated
  around or above G ~ 0.015 -- the C++ full solver's own stall floor (exp07 traces) is at
  the same level as the compressed representation floor, so no side of any earlier C++
  comparison ever measured below the plateau;
- the earlier near-zero reported differences were OBJECTIVE differences relative to the
  chosen baseline, not evidence of a near-zero common full-space equilibrium gap;
- therefore say: "3.26x with a 0.046% Beckmann-objective difference in the tested
  moderate-equilibrium-accuracy regime" -- NOT "3.26x with essentially zero accuracy loss"
  -- and report the common-gap limitation separately.

Accuracy-runtime envelope (K10, matched-ALM, this engine): at G >~ 0.015 compressed
iterations provide a meaningful computational advantage (~2.4-2.7x per outer); below
G ~ 0.014 the tested global-SVD representation cannot reach the required accuracy at any
tested rank.

## 6. Relation to the campaign's other certified findings

- **w0 is the same geometry seen from the other side** (W0_LOAD_BEARING.md): the anchor
  places the start in the box interior; the floor occurs where the trajectory reaches the
  box boundary. Both are statements about span(U) meeting {x >= 0}.
- **The grouped/box-bound representation** (highk_fixes: gap 0.0000 on K10, best objective
  of any operator, in Python) does not have this floor mechanism -- its variables are
  nonnegative combination weights, so feasibility is a plain box bound and no signed
  reconstruction crosses zero. It was 0.54x on speed in Python; whether its accuracy
  advantage survives a compiled implementation is the open Paper-2-adjacent question.
- **The reduction ceiling** (exp06) is unaffected: it is a counting argument and holds for
  any representation that keeps one explicit path per OD.

## 7. Reproduction

    # ten synchronized steps + kernel split (writes exp08_lockstep_K10_d05_r6.csv)
    python drivers/exp08_lockstep.py --case K10 --scenario d05 --outers 10 --rank 6
    # the five controls from the k=4 branch state (writes exp09_controls_K10_d05.csv)
    python drivers/exp09_controls.py --case K10 --scenario d05 --branch-k 4 --rank 6
    # compact generalization check (C3/C4 only) on another pool
    python drivers/exp09_controls.py --case K15 --scenario d05 --controls C3,C4

Determinism caveat: ARPACK's random start vector makes U reproducible only up to column
signs across processes, so exp09 re-runs the lockstep to the branch point in-process
rather than decoding exp08's checkpoints; reproduction G values match exp08 to ~3e-4
(e.g. k=4: 0.013523 vs 0.013580). Seeds and every setting are in the drivers and
`config.py`. exp08 checkpoints (ck_*.npz) live under %TMP%/lockstep/ and are valid only
within the process that wrote them.

## 8. Status of the generalization check

C3/C4 on Chicago Sketch K15 (d05, seed 100) launched 2026-07-27; result to be appended to
`results/exp09_controls_K15_d05.csv`. The scenario sweep (exp07 d10/c05/c10) is
deliberately NOT being restarted.
