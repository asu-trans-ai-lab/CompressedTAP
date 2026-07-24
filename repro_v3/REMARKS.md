# Remarks for reviewers of the v3 reproduction

Points a second reader should check independently before any `[fill]` cell is trusted.
Each remark states what was found, how it was established, what was ruled out, and what
it changes. Everything here is reproducible from this tree; producing scripts are named.

Status: R1–R3 and R7–R8 established and acted on; R4–R6 are open items that need an
author ruling or further work. Nothing in the manuscript has been edited on the basis of these remarks.

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

## R8. The compressed rows stall at a shared certificate — a representation limit

**Finding (Sioux SFK25, tau=600 at the 90th percentile, r=50).** Two operators with
nothing in common but the representation stop at the same place:

| representation | operator | certificate | iterations | CPU | `Gap_F` | `delta_F` |
|---|---|---|---|---|---|---|
| full | GP [reference] | 7.55e-10 | 186 | 1.07 s | +0.000000% | 0.0000% |
| full | GP | 7.81e-07 | 89 | 1.55 s | +0.000000% | 0.0000% |
| full | Frank-Wolfe | 9.99e-07 | 96,758 | 162.25 s | +0.000116% | 0.0000% |
| full | ALM | 1.14e-03 | 1,031 | 2.11 s | +0.048187% | 0.0000% |
| **compressed** | **ALM** | **2.99e-02** | 258 | 0.49 s | **+3.490901%** | 0.0000% |
| **compressed** | **GP-signed (penalised)** | **2.91e-02** | 6,303 | 600 s (capped) | **+3.460673%** | 0.1031% |

The full-representation operators reach 1e-6..1e-10. Both compressed operators stall
three orders of magnitude short, at the *same* certificate and within 0.03 percentage
points of the same objective gap — despite one using multipliers plus penalty and the
other Dykstra projection plus penalty. **That signature is a property of the feasible
set, not of either solver:** at this threshold and rank the signed-SVD representation
simply cannot get nearer the optimum.

**Why this matters for Table 2.** The table's stated purpose is to show "the compressed
formulation is not intrinsically tied to the AL operator". These two rows support that
directly: change the operator entirely and the compressed optimum does not move.

**A symmetry worth noting.** On the full representation GP dominates (1.55 s vs ALM's
2.11 s and FW's 162 s); on the compressed representation ALM dominates by three orders of
magnitude (0.49 s / 258 iterations vs 600 s / 6,303 iterations). This is consistent with
the manuscript's framing that the AL method is designed for the compressed structure,
whose dense constraint it carries through multipliers rather than through projection.

**Caveat.** `GP-signed` hit its 600 s cap rather than converging, so slow further progress
cannot be strictly excluded — though it had already passed ALM's certificate (2.91e-02 vs
2.99e-02) when it stopped. Its `delta_F = 0.1031%` (the only nonzero one in the table) is
the expected trace of carrying the dense reconstructed nonnegativity by penalty: the
terminal iterate retains small negative minor flows that the conversion then removes.

**Implementation flaw to fix before Panel B.** `gp_compressed.py` evaluates the shared
certificate on *every* iteration, and that evaluation rebuilds the full path vector
(a 5,931x50 product) and runs a 528-OD Python projection loop. Most of the 600 s went
there rather than into the optimisation. The certificate should be checked every ~25
iterations. This does not affect the plateau above — that is set by the representation —
but it does make the reported CPU an overstatement of the operator's intrinsic cost, and
Panel B's GP speedup column must not be produced until it is fixed.

---

## R9. Two Python loops over OD pairs dominated the run time; the fix is not "vectorise it"

Both hot inner routines looped over OD pairs in Python and were called once per operator
iteration. On Chicago Sketch (**709,567 paths, 82,083 OD pairs, 2,950 links**) that is
82,083 interpreter iterations per iteration of every solver.

| routine | before | after | check |
|---|---|---|---|
| `v3_metrics.convert_euclid` (per-OD simplex projection) | per-OD Python loop | one lexsort + segmented cumulative sum | matches loop to **5.8e-11** across perturbation scales with up to 49% negative entries; **6.9x** faster on Sioux |
| `run_table2_consistency.LMOIndex` (linear minimisation oracle) | per-OD Python loop | grouping sorted **once** at construction, then two C-level `reduceat` reductions per call | matches loop **exactly (0.0e+00)**; **18.6x** faster on Sioux |

**The lesson worth recording.** The obvious fix for the oracle — a `lexsort` inside every
call — was implemented first and measured *slower than the Python loop* on Sioux (0.7x),
because sorting 6,464 elements every call costs more than 528 small `argmin` calls. What
made it fast was noticing that **the grouping never changes between iterations**, so the
sort belongs in the constructor and only the segmented reduction belongs in the call.
Both speedups grow with the OD count, so the margin on Sketch and Regional is much larger
than the Sioux figures above.

**Numerical caveat carried with the projection.** The segmented cumulative sum forms its
per-block sums by subtracting one global cumulative sum from another. On very large pools
(Regional: 4.8M paths, running sums of order 1e10) cancellation can cost significant
digits. An O(n) guard verifies the result lands on the demand constraint and falls back to
the exact per-block routine otherwise. It has not fired on any instance tested; a reader
running Regional should confirm it still does not.

---

## Gate status

| gate | meaning | status |
|---|---|---|
| 0 | metric module correct (7 checks) | PASS |
| 1 | reference self-evaluates to `Gap_F = 0`, `R^2 = 1` | PASS (Sioux) |
| 2 | no negative `Gap_F` anywhere | **PASS** (Sioux, GP reference; all six rows positive) |
| 3 | reported identities consistent (`K/q` vs `n`, `s+r`) | not yet reached |
| 4 | every rendered LaTeX row traces to a CSV row | not yet reached |
