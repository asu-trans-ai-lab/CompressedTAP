# Remarks for reviewers of the v3 reproduction

Points a second reader should check independently before any `[fill]` cell is trusted.
Each remark states what was found, how it was established, what was ruled out, and what
it changes. Everything here is reproducible from this tree; producing scripts are named.

Status: R1, R3, R7–R9, R11–R12 established and acted on; R2 superseded by R10; R4–R6
and R10's remedy are open items needing an author ruling or further work. Nothing in the manuscript has been edited on the basis of these remarks.

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

## R10. The reference problem is scale-dependent: no fixed operator works at both sizes

**This supersedes the fix recorded in R2, which is correct on small instances only.**

R2 replaced the tau=0 ALM reference with full-path gradient projection, because on Sioux
Falls ALM terminated 0.048% above the optimum. Running the same driver on Chicago Sketch
(**709,567 paths, 82,083 OD pairs, 2,950 links**) inverts the result:

| instance | GP reference | ALM | verdict |
|---|---|---|---|
| Sioux SFK25 (6,464 paths) | cert **7.55e-10**, 186 it, 1.07 s | cert 1.14e-03, `Gap_F` **+0.048187%** | ALM is the bad reference |
| Chicago Sketch (709,567 paths) | cert **3.75e-03**, 920 it, **1200.9 s (hit cap)** | 758.7 s, `Gap_F` **-0.028254%** | **GP is the bad reference** |

On Sketch the gradient-projection reference does not converge inside twenty minutes, and
ALM finds a better feasible point — so `Gap_F` goes negative again and Gate 2 fails, for
the opposite reason to Sioux. Six orders of magnitude of certificate quality separate the
same operator on the two instances (7.55e-10 vs 3.75e-03).

**Conclusion.** Reference quality is a property of the instance, not of the operator.
Fixing any single solver as `v^ref` will fail somewhere in a three-network study.

**Proposed remedy (author decision pending).** Define `v^ref` as *the best feasible point
found by any uncompressed operator under the common protocol*, and record its provenance
and certificate as columns in every CSV. This (i) makes `Gap_F >= 0` hold by construction
rather than by luck, which is the property the v3 revision exists to establish;
(ii) satisfies the manuscript's wording, which asks for "a consistently converged
uncompressed formulation" without naming a solver; (iii) costs nothing, since those
solutions are computed for the table anyway; and (iv) stays auditable.

**The cost that must be stated, not hidden.** On large instances the best available
reference is only converged to ~1e-3, so **the resolution of the `Gap_F` column is no
better than that** on those networks. Any compressed row whose true gap is smaller than
the reference's own suboptimality cannot be distinguished from zero. This limitation
belongs in the manuscript, not just in this file.

**Measured cost, for budgeting (not an estimate).** On Chicago Sketch: one ALM solve
758.7 s; the GP reference still at 3.75e-03 after 1200 s. Extrapolating the observed
1.3 s/iteration, reaching 1e-8 would take roughly 3.6 hours on this network alone.
Chicago Regional has 4.8M pool rows, about 6.8x Sketch.

**A driver gap this exposed.** `op_gp_full` and `op_fw_full` take `max_seconds`;
`ca.solve_full` and `ca.solve_compressed` do not. ALM therefore cannot be time-capped in
Panel A, where it runs 4 thresholds x 3 repetitions x 3 networks. Needs a wall-clock
guard before that campaign starts.

---

## R11. A real bug: the BB step is nonmonotone and the drivers returned the last iterate

**Symptom on Chicago Sketch.** Every full-representation operator beat the reference, and
one of them was the *same operator run longer*:

| row | `Gap_F` | CPU | note |
|---|---|---|---|
| GP, used as reference | +0.000000% | 1200.9 s (cap) | certificate 3.75e-03 |
| ALM | -0.028254% | 758.7 s | |
| Frank-Wolfe | -0.057411% | 600.1 s (cap) | |
| **Gradient projection** | **-0.054293%** | **601.5 s (cap)** | **same operator, half the time, better point** |

A 600 s GP run producing a better point than a 1200 s GP run cannot be a convergence
story. The cause is that the Barzilai-Borwein step is **deliberately nonmonotone** -- the
objective is allowed to rise before it falls, which is where its speed comes from -- and
the driver returned the *last* iterate rather than the best one seen. Running longer
therefore does not guarantee a better answer.

**Fix.** `op_gp_full` and `op_fw_full` now track the best iterate by certificate and
return that. (FW's step is monotone in the objective, but the wall-clock cap can stop it
anywhere, so it gets the same treatment.) This is standard practice for nonmonotone
methods and should have been there from the start.

**What it invalidates.** The Sketch rows above, and any timing comparison drawn from
them. The Sioux rows are unaffected in substance -- GP converged to 7.55e-10 there, so
best and last coincide -- but they will be regenerated under the fixed code anyway.

**Caution for the reader.** This bug and R10 point the same way and could be confused. R10
is real and independent: even with best-iterate tracking, GP on Sketch reached only
3.75e-03 in twenty minutes against 7.55e-10 in one second on Sioux. Fixing R11 removes
the absurdity of longer-is-worse; it does not make a fixed operator a safe reference.

---

## R12. `delta_F` earns its place: 36.4% on the compressed Sketch row

The compressed ALM row on Chicago Sketch (tau = 1.01 at the 90th percentile, r = 50,
118,174 majors of 709,567 paths) reports

    delta_F = 36.3803%,   Gap_F = +0.231375%,   link difference 2.0712%

against `delta_F = 0.0000%` for every full-representation row and for the compressed row
on Sioux. The common feasibility conversion is moving **over a third of the L1 path-flow
mass** before the objective is evaluated.

This is exactly what the v3 metric was introduced to expose. The manuscript's own
justification -- "the added column `delta_F` makes the size of the conversion visible
rather than hiding it in the evaluation procedure" -- is vindicated on the first large
instance tried: a reader seeing only `Gap_F = +0.23%` would conclude the compressed
solution was close, without knowing how much repair produced that number.

Two implications. First, `delta_F` must be reported in Panel A for every threshold, not
only where it happens to be small. Second, a `Gap_F` obtained after a 36% conversion is a
statement about the *converted* point, not about what the compressed solver returned, and
the text should say so.

---

## R13. Large-network reference is ~1e-3; FW skipped on large nets; wall-clock guards added

Measured on Chicago Regional (**4,829,025 paths, 297,900 OD, 39,018 links**, 148 s to load,
~4 GB resident): full-path gradient projection reached certificate **1.562e-03** in **277
iterations over 1805 s** (30-min cap) -- about 6.5 s per iteration, six orders of magnitude
looser than the 7.13e-08 it reaches on Sioux in 0.2 s. The AL candidate did not converge
in 49 minutes and had no wall-clock exit.

Three author decisions (2026-07-19) follow, all now in the drivers:

1. **Frank-Wolfe is skipped as a reference candidate on the large networks** (Sketch,
   Regional, Philadelphia). It needs ~1e5 iterations even on Sketch (56,607 on Sioux to
   reach 1.6e-06) and is never the best feasible point. This is efficiency only: v^ref
   remains the best point among the operators that run, and `ref_operators` records which
   ran on every row.
2. **Every reference candidate shares one wall-clock cap** (`--ref-cap`, default 900 s),
   and `solve_full` / `solve_compressed` now take a purely-additive `max_seconds` guard
   (default None = unchanged) checked at each outer-iteration boundary, so ALM is capped
   too. Per-threshold solves share `--solve-cap` (default 1800 s). No candidate or solve
   can hang the campaign.
3. **Panel A runs unattended, large networks first** (`run_panelA_overnight.py`):
   regional -> philadelphia -> sketch -> sioux, each in its own subprocess with isolated
   memory, tee'd to `results/logs/panelA_<net>.log`, with a per-network ceiling.

**Consequence carried into the manuscript (decision A already requires it):** on the large
networks the reference converges only to ~1e-3, so the `Gap_F` column's resolution is no
better than that there. A compressed row whose true gap is below the reference's own
suboptimality is indistinguishable from zero. `ref_cert` and `ref_obj_spread_pct` on every
row make this visible.

**Validation.** The full chain passed on Sioux (best-of-all reference picks GP over ALM
and FW, `Gap_F` positive on all four rows, Gate 2 PASS, candidate spread 0.048% recorded).

---

## R14. The wall-clock guard is checked at outer-iteration boundaries only

`solve_full`/`solve_compressed` check `max_seconds` at each OUTER iteration boundary, not
inside the inner L-BFGS-B solve. On huge networks a single outer iteration can itself
exceed the cap: 200 inner iterations x a 4.8M-path matvec at ~6 s each is ~1000-1200 s,
so on Chicago Regional the ALM reference candidate ran past its 900 s cap and only stopped
after completing the first outer iteration (~21 min observed).

**Consequence, not a bug.** The effective ceiling for ALM on the large networks is
"cap + one outer iteration", which can be roughly 2x the nominal cap. The run still stops;
the campaign does not hang. But a reader must not read `--ref-cap 900` as "ALM ran for at
most 900 s" -- the actual candidate time is recorded per row and should be read from there,
not inferred from the cap.

The GP and FW operators do not have this issue: their loops check the clock every
iteration, so their caps are tight (GP stopped at 905.9 s on Regional). Only the two AL
solvers, whose inner work is a single opaque L-BFGS-B call, overrun by up to one outer
iteration.

If a tighter ALM ceiling is ever needed, lowering `maxiter_inner` bounds the overrun
directly (fewer inner iterations per outer); it was left at the certified default here so
the reference candidate is the same AL method the paper describes.

---

## R15. Best-of-all reference is only well-defined among CONVERGED candidates

On Chicago Regional the tau=0 row came out **negative**: `Gap_F = -0.08537%`. This is not
the R2/R10 fixed-operator problem and not a bug in the metric; it is a consequence of
comparing two differently-truncated runs of the same AL method.

- The reference candidate `ALM-full` was stopped by the 900 s guard after 400 inner
  iterations (t = 1349 s with the R14 outer overrun), objective 19,132,505.42.
- The tau=0 threshold solve is *also* an uncompressed AL solve, but under the 1800 s
  `solve-cap` it ran 600 inner iterations (t = 1925 s) and reached a better objective, so
  its `Gap_F` against the reference is negative.

**Root cause.** On the large networks no candidate converges; every one is truncated by a
wall-clock cap. "Best feasible point among the candidates" is then only well-defined if the
candidates are compared at the *same* budget. The reference candidate and the tau=0 solve
have different budgets (900 s vs 1800 s), so the longer one wins and the sign flips.

**Scope.** This can only affect the tau=0 row, which is itself an uncompressed solve of the
same kind as the reference. Compressed rows (tau>0) solve a different, smaller problem and
are not expected to beat a full-length uncompressed reference; they must still be checked.

**Fix options (author decision at analysis time, not mid-run):**
1. Make the tau=0 threshold solve BE the reference on large networks (it is the
   best-converged uncompressed solve available), and drop the separate reference candidate
   there. Cleanest: one uncompressed solve, used both as the tau=0 row and as v^ref.
2. Give every uncompressed candidate the same budget as the tau=0 solve (raise `ref-cap`
   to `solve-cap`), so the reference is at least as converged as any uncompressed row.
3. After the run, post-process: set v^ref to the min objective over ALL uncompressed
   feasible points produced anywhere in the campaign (reference candidates AND the tau=0
   row), then recompute every Gap_F. This uses the data already on disk and needs no rerun.

I recommend option 3 for the current overnight data (no recompute cost) and option 1 for
any future run. The overnight run is NOT interrupted; the raw objectives are all recorded,
so Gap_F can be recomputed against the true best point without re-solving.

---

## R16. The overnight Panel A config is mis-calibrated for 4.8M-path networks

Discovered by running, not estimable in advance. On Chicago Regional (4,829,025 paths)
each per-threshold cost is far higher than planned, from three compounding effects:

1. **A single ALM/compressed outer iteration is 20-40 min** on this network (maxiter_inner
   =200 inner L-BFGS-B steps, each a full sparse matvec plus a 1.16M x 50 dense U@z at
   tau=0.23). The `solve-cap` guard is checked at the OUTER boundary (R14), so it cannot
   bound the work below one outer -- a 1800 s cap does not stop a 2400 s outer.
2. **Each compressed threshold runs TWO solves**, the main one and the w0-free diagnostic
   (decision 3). On small networks the diagnostic is free; on Regional it doubles the
   per-threshold time to ~60-80 min. tau=0.23 was still running at 54 min for this reason.
3. **The 150-min net-ceiling is below the ~4-5 h a large network actually needs**
   (reference phase ~40 min + 4 thresholds x ~60-80 min). Regional is killed before
   finishing; the tau=0 row and reference are salvaged from the log to
   `results/table4A_regional_salvaged.csv`.

**Nothing here is a correctness bug** -- the numbers produced are valid, and Gap_F on the
tau=0 row is negative only because of the truncated-reference issue of R15, fixable by
recompute. The problem is purely that the run cannot complete the large networks in one
150-min slot.

**Recalibrated plan (for author approval; not executed autonomously because two options
touch the certified solver):**
- (a) On large networks use the tau=0 full solve directly as v^ref (R15 option 1), dropping
  the separate reference phase -- saves ~40 min/network and removes the R15 sign flip.
- (b) Lower `maxiter_inner` to ~50 on large networks so each outer is ~4x shorter and the
  wall-clock guard becomes effective. This departs from the certified default (200) and
  needs approval.
- (c) Skip or cheapen the w0-free diagnostic on large networks (it is a CSV-only diagnostic
  per decision 3; halving the per-threshold cost may be worth losing it on the big nets).
- (d) Raise the net-ceiling to ~5 h/network and run one large network per night.

Incremental CSV (committed 4f37cd3) already protects philadelphia and sketch: whatever
thresholds they complete before the ceiling are saved, unlike regional's all-or-nothing
prior-code run.

---

## R17. C++ replaces Python for Gap_F and R2, but delta_F is solver-terminal-dependent

The certified C++ compressed_solver is the same SPG-ALM signed-SVD algorithm and ~8-28x
faster (R16), which is the only practical way to finish the 4.8M-path networks. Before
trusting it in a table, `validate_cpp_metrics.py` ran the Python solve on the IDENTICAL
(P, C) instance (Chicago Sketch, tau=1.06, r=50) and computed all v3 metrics on both
terminal iterates through the one shared v3_metrics module, against a common GP reference
(f_ref=16,771,252, cert 8.86e-06).

| metric | Python solve | C++ solve | |Delta| |
|---|---|---|---|
| Gap_F | +0.297% | +0.604% | **0.31 pp** |
| R2 | 0.999188 | 0.998001 | **0.0012** |
| delta_F | **38.49%** | **19.14%** | **19.36 pp** |

**Gap_F and R2 are consistent** -- both are properties of the feasible-projected link
flows, which the two solvers reach within 0.3 pp / 1e-3 of each other. C++ can replace
Python for the substantive accuracy columns without changing what they mean.

**delta_F is NOT consistent, and the reason matters:** delta_F is the L1 mass the
feasibility conversion moves, i.e. a property of the solver's UNPROJECTED terminal iterate,
not of the representation. Python's ALM stops with 38% of the mass still infeasible; C++'s
hard-ALM stops at 19% (a more feasible terminal point). Both are valid; they are different
numbers because the two solvers terminate at different raw points.

**Consequence.** Switching Panel A to C++ changes the delta_F column. That is acceptable
only if (a) C++ is used for EVERY row so the column is internally consistent, and (b) the
manuscript states delta_F is the conversion size for the reported solver, not a property of
the compression. This actually sharpens R12: delta_F measures how much repair a given
solver's terminal point needs, which is solver-dependent by definition.

**Incidental, useful:** the GP reference reached cert 8.86e-06 in 600 s on Sketch here,
far better than the 3.75e-03 seen before the R11 best-iterate fix. The large-network
reference is therefore in better shape than R13 feared, because R11 made GP return its best
iterate rather than a nonmonotone-BB wandering endpoint.

**Validator caveat:** its VERDICT only checked Gap_F and R2 (it passed); delta_F was
outside the pass condition and must be read from the table above, not from the verdict.

---

## R18 [RESOLVED 2026-07-24, author]: use the submitted tau VALUES + the RECOMPUTED reduction

Author decision: the revnote requires preserving the submitted threshold GRID (the tau
values), and decision 2 recomputes the reduction column. So Panel A reports the original
tau labels (0/0.23/0.46/1.03 etc.) with the reduction RE-COMPUTED from the current
split_major_minor -- e.g. regional tau=0.23 -> 53.8%, not the submitted 23.9%. This is
compliant: the tau grid is preserved, the reduction is a rerun output. No tau remapping.
The difference from the submitted reductions reflects the rerun's nominal-flow split and is
stated as such. Original investigation note follows.

### Original note

The C++ Panel A run gives, for Chicago Regional:
  tau=0.23 -> reduction 53.8% (GapF +2.18%, R2 0.928, delta_F 89.7%)
The submitted manuscript grid says regional tau=0.23 -> 23.9%, and 53.8% is its tau=1.03
value. So `split_major_minor(P, tau)` at tau=0.23 produces the split the manuscript
attributes to tau=1.03 -- an apparent tau-scale mismatch between the certified loader's
absolute-flow threshold and the manuscript's tau values.

This is NOT a C++ artifact: the Python driver passes the same taus to the same
split_major_minor. It affects the reduction column (and hence which compressed problem is
solved at each labelled tau) on all networks. Two possibilities: (a) the manuscript tau is
a quantile or scaled quantity, not an absolute nominal-flow threshold; (b) the nominal
flow x0 used now differs from the submitted run. Must be resolved before the table is
final, since 'preserve the submitted threshold grid' requires the labelled tau to select
the same split. Flagged mid-run; the numbers produced are internally valid (Gate 2 holds),
but the tau labels may need remapping. Investigate against the submitted appendix
reductions after the run completes.

---

## Gate status

| gate | meaning | status |
|---|---|---|
| 0 | metric module correct (7 checks) | PASS |
| 1 | reference self-evaluates to `Gap_F = 0`, `R^2 = 1` | PASS (Sioux) |
| 2 | no negative `Gap_F` anywhere | **PASS** on Sioux; **FAILS on Chicago Sketch** (ALM -0.028%, the GP reference being the under-converged one there) -- see R10 |
| 3 | reported identities consistent (`K/q` vs `n`, `s+r`) | not yet reached |
| 4 | every rendered LaTeX row traces to a CSV row | not yet reached |
