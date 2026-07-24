# Remarks for reviewers of the v3 reproduction

Points a second reader should check independently before any `[fill]` cell is trusted.
Each remark states what was found, how it was established, what was ruled out, and what
it changes. Everything here is reproducible from this tree; producing scripts are named.

Status: R1–R3 established and acted on; R4–R6 are open items that need an author ruling
or further work. Nothing in the manuscript has been edited on the basis of these remarks.

---

## R1. `P_X` is ambiguous, and the ambiguity is *not* harmless in the way one expects

**Where.** Manuscript Eq. (metric): `x^F = P_X(x~)`, "performed OD pair by OD pair over
`X = {x >= 0, Ax = d}`". `P_X` conventionally denotes Euclidean projection. The certified
pipeline's `project_feasible` instead does clip-and-rescale (`max(x,0) * d/s`). Both land
in `X`; they are different maps.

**Finding (a) — `delta_F` is provably map-independent.** Per OD, with
`s = sum_{x>=0} x` and mass to shed:

| map | L1 displacement |
|---|---|
| clip-and-rescale | `sum_{x<0}|x| + (s - d)` |
| Euclidean `(x - theta)_+` | clipped terms give `sum |x_j|`, kept terms give `theta` each, and `k*theta = sum_S x - d`, summing to `sum_{x<0}|x| + (s - d)` |

The two are algebraically equal. Confirmed numerically to 1e-8 in
`test_v3_metrics.py` check [6], including a case with 35% negative entries. **So the
`delta_F` column cannot be corrupted by this choice** — a reader may take it at face
value regardless of which map an implementer used.

**Finding (b) — `Gap_F` is strongly map-dependent.** Same iterate, same reference:
Euclidean `+12.62%`, clip-and-rescale `-0.63%`. The rescale map can therefore produce a
*negative* feasible gap, which is precisely the artifact the v3 revision exists to remove.

**Consequence.** This campaign uses the Euclidean projection for every paper number, and
carries the rescale value in `rescale_*` diagnostic columns so the choice stays visible
rather than buried. **Check:** any independent reproduction that uses `project_feasible`
will not match the `Gap_F` column, and may see negative gaps. That is expected, not a
discrepancy.

*Producing script:* `python/test_v3_metrics.py` (Gate 0, 7/7).

---

## R2. The `tau = 0` ALM run is not a valid reference

**Where.** Manuscript: "The revised experiments use a consistently converged uncompressed
formulation as the primary reference"; Panel A takes the `tau = 0` **ALM** run as `v^ref`.

**Finding.** On Sioux Falls SFK25 (6,464 paths, 528 OD), measured against a common
optimality certificate (relative Frank-Wolfe duality gap of the converted iterate):

| solution | certificate | objective | vs best |
|---|---|---|---|
| ALM `tol=1e-8`, `maxiter_inner=200` (certified default) | 1.14e-03 | 4,233,374.255 | **+0.0482 %** |
| ALM `tol=1e-8`, `maxiter_inner=2000` | 7.53e-04 | 4,232,323.685 | +0.0234 % |
| Frank-Wolfe `tol=1e-6` (96,758 it, 219.4 s) | 9.99e-07 | — | ~+0.000 % |
| **full-path GP `tol=1e-9` (186 it, 2.05 s)** | **7.55e-10** | **4,231,335.287** | **0** |

**Ruled out.**
- *Infeasibility.* Every method converts with `delta_F = 0.0000%` and demand residual
  1e-12..1e-13. The conversion does no work here.
- *The outer tolerance.* ALM's `tol` tests `||Ax - d|| < tol` — a **feasibility**
  tolerance that says nothing about optimality. Tightening 1e-6 -> 1e-12 (CPU 2.55 s ->
  11.34 s) leaves the certificate pinned at 1.1355e-03 and moves the objective only in
  the 10th significant digit.
- *The inner budget alone.* `maxiter_inner` 200 -> 2000 improves the certificate to
  7.53e-04 and the objective by 0.025%; 2000 -> 20000 changes nothing. A second plateau
  remains. `gtol` is already set to `tol * 0.1`.
- *A single misbehaving solver.* FW and GP converge independently to the same point
  (`Gap_F` -0.048048% vs -0.048164%, agreeing to 1.2e-4 percentage points) and both
  differ from ALM in the same direction.

**Consequence.** With an ALM reference, every `Gap_F` in Panel A, Panel B and Table 5 is
biased low by roughly 0.05%, and any compressed run landing nearer the optimum reports a
negative gap. Author decision (2026-07-19): take `v^ref` from full-path gradient
projection. It is still the uncompressed formulation — so the manuscript's wording holds —
is six orders of magnitude better converged, and is *cheaper* (2.05 s vs 5.21 s). ALM
becomes one of the compared operators in Table 2, evaluated against that reference like
the others. After the switch, ALM's `Gap_F` reads `+0.048187%` rather than a negative
number.

**Check:** re-run `python/run_table2_consistency.py --net sioux` and confirm the
reference self-evaluates to `Gap_F = 0.000000%`, `R^2 = 1`, and that no row is negative.

*Producing script:* `python/run_table2_consistency.py`; detail in
`results/GATE2_REFERENCE_FINDING.md`.

---

## R3. `GP-signed` is not an exact projection, and is labelled accordingly

**Where.** Decision 1(a): run a projection operator on v3's own signed-SVD representation
so the compressed rows are not produced by the AL operator alone.

**Finding.** The compressed feasible set

    A1 y + M z = d_eff,   y >= 0,   w0 + U_r z >= 0

admits no closed-form projection: the equality couples every OD through the *shared*
latent block `z`, and the last constraint is dense (one row per minor path). This is the
manuscript's own point about the signed decoder, now met in the solver design.

**What was implemented.** `python/gp_compressed.py`:
- the equality-plus-bound part `{A1 y + M z = d_eff, y >= 0}` is projected **exactly**,
  by Dykstra alternation between the affine set (one prefactorised Cholesky least-norm
  correction) and the bound;
- the dense reconstructed nonnegativity is carried by the **same quadratic penalty the AL
  formulation uses**, with the weight held fixed during the projection.

**Consequence.** The operator is reported everywhere as
`GP-signed (dense constraint penalised)` and must never be described as an exact
projection onto the compressed set. A reader comparing it against `GP-grouped` — where
the feasible set *is* a product of simplices and the projection is exact and closed-form —
is comparing two genuinely different geometries, which is why decision 1 requires both to
be reported and never merged into one "GP" column.

---

## R4. An incidental operator result, reported but not yet claimed

Full-path gradient projection reached the same 1e-6 duality-gap certificate as
Frank-Wolfe in **89 iterations / 1.18 s** versus FW's **96,758 iterations / 219.4 s** —
a factor of 186 in wall time on this instance. This agrees independently with the LPG
result recorded in the v3.1 portability package. It is **diagnostic-class evidence on one
network** and is not proposed for any table; it is noted because it bears on how
"matched stopping tolerances" should be read in Table 2, whose caption already omits CPU.

---

## R5. Open: `x_raw` was added to the certified solvers

`solve_full` and `solve_compressed` now return an extra key `x_raw`, the **unprojected**
terminal iterate. The v3 metrics need it: the raw objective difference, `||A x~ - d||_inf`
and `delta_F` are all defined at `x~` *before* conversion, and the previous return value
had already been passed through `project_feasible`.

The change is purely additive — no existing key, value or computation is altered, so any
prior result remains bit-identical. **Check:** `git diff` on
`source/updated_TAPLite/python/compressed_assignment.py` should show only the two added
dict entries and their comment.

---

## R6. Open: the affine offset `w0` is on v3's critical path

v3's representation is `w0 + U_r z`. A controlled ablation of 2026-07-19 (toggling only
`w0`, holding basis, rank, solver, constraints and reference fixed) found the offset to
be the dominant error source in support-sparse regimes: on all 24 calibrated real
subproblems the offset-free variant was numerically exact (1e-8..1e-11) where the affine
form gapped 2.9–190%; only under spread support (demand x4) was the affine form
competitive (7/16 vs 9/16).

Per decision 3 the offset-free variant is carried as a **diagnostic column in the CSVs
only** and does not enter the manuscript, since the revnote restricts this round to a
numerical rerun. Flagged here so a reader is not surprised to find the column, and so the
question is not lost.

*Evidence:* `results_certified/reproduction/w0_ablation/` on branch `major_latent_dev`.

---

## R7. A shape bug in the first `GP-signed` draft, and the unit check that caught it

`build_compressed` returns `D = B2' U` with shape (links x rank) and `M = A2 U` with
shape (OD x rank). The first draft of `gp_compressed.py` used `D' z` for the link flow
and `D t` for the latent gradient — both transposed. On Sioux (D is 76x50) this raised a
dimension error rather than producing wrong numbers, but on a network where the two
dimensions happened to match it would have failed silently.

The check now in place, which any reader should re-run after touching that file:

    link flow via the compressed formula   v_base + B1' y + D z
    link flow via explicit reconstruction  v0 + B'(x with x[minor] = w0 + U z)
    -> agree to 3.6e-11
    projector output: equality residual 1.8e-12, min major flow > 0

*Producing snippet:* see the unit check invoked before the Sioux rerun; shapes on that
instance are `B1 (533,76)`, `D (76,50)`, `M (528,50)`, `U (5931,50)`, `s=533`, `r=50`.

---

## Gate status

| gate | meaning | status |
|---|---|---|
| 0 | metric module correct (7 checks) | PASS |
| 1 | reference self-evaluates to `Gap_F = 0`, `R^2 = 1` | PASS (Sioux) |
| 2 | no negative `Gap_F` anywhere | FAILED with ALM reference -> PASS expected with GP reference (rerun in progress) |
| 3 | reported identities consistent (`K/q` vs `n`, `s+r`) | not yet reached |
| 4 | every rendered LaTeX row traces to a CSV row | not yet reached |
