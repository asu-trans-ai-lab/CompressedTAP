# v3 fill-cell specification

Every `[fill]` cell in `main.tex` (30 of them), the command that produces it, and the CSV
column it reads. No number is ever typed by hand: `fill_tables.py` renders the LaTeX rows
from `results/*.csv`.

Author decisions of 2026-07-19 governing this campaign:

1. **Both gradient-projection variants are produced and labelled separately.**
   - `GP-signed` — projection over `{A1 y + M z = d_eff, y >= 0}` with the dense
     reconstructed constraint `w0 + U_r z >= 0` carried by an active set. Keeps v3's
     signed-SVD representation exactly as the manuscript defines it.
   - `GP-grouped` — projection over a product of demand simplices on a grouped
     (nonnegative, convex-combination) representation, where the projection is exact and
     closed-form.
   Both are reported; they answer different questions and must never be merged into one
   "GP" column.
2. **Everything is re-timed.** No submitted CPU or iteration counts are reused. Each row's
   timing and its `Gap_F` come from the same fresh run, so no row mixes two campaigns.
3. **The offset-free variant (`w0 = 0`) is carried as a diagnostic column in the CSVs
   only.** It does not enter the manuscript. Rationale: v3's representation is
   `w0 + U_r z`, and the 2026-07-19 ablation found the affine offset to be the dominant
   error source in support-sparse regimes; recording the variant costs nothing on the same
   solve path and preserves the option without adding a benchmark the revnote forbids.

4. ~~**`v^ref` is taken from full-path gradient projection**~~ **SUPERSEDED by decision A
   below.** Retained for the record because the measurement behind it is still valid:
   (added 2026-07-19 after Gate 2 failed; see `results/GATE2_REFERENCE_FINDING.md`).
   Measured on Sioux SFK25, the certified ALM default terminates with an optimality
   certificate of 1.14e-03 and an objective 0.0482% above the optimum, and cannot be
   driven closer through its exposed tolerances -- `tol` is a *feasibility* tolerance
   (`||Ax-d|| < tol`) and tightening it 1e-6 -> 1e-12 leaves the certificate unchanged;
   raising `maxiter_inner` 200 -> 2000 helps once (to 7.53e-04) and then plateaus again.
   Full-path GP reaches 7.55e-10 in 186 iterations and 2.05 s -- six orders of magnitude
   tighter and cheaper than ALM. It is still the uncompressed formulation, so the
   manuscript's wording ("a consistently converged uncompressed formulation as the
   primary reference") is satisfied. ALM appears in Table 2 as one of the compared
   operators, evaluated against that reference like the others.

### Decision A (2026-07-19) — `v^ref` is the best feasible point from ANY uncompressed operator

Reference quality is a property of the instance, not of the operator: on Sioux the AL run
was 0.048% above the optimum while GP reached 7.55e-10 in one second; on Chicago Sketch GP
reached only 3.75e-03 in twenty minutes and AL found a better point (R10). Fixing either
produces a negative `Gap_F` somewhere in a three-network study.

Every uncompressed operator is therefore run, each terminal iterate is converted to a
feasible point **by the per-OD simplex projection — demand equality and nonnegativity
together — and only then compared**, and the lowest feasible objective becomes `v^ref`.
`Gap_F >= 0` holds for every uncompressed row by construction. Implemented in
`python/v3_reference.py`; `ref_source`, `ref_cert` and `ref_obj_spread_pct` are columns on
every row.

**Cost that goes in the manuscript, not only here:** on large instances the best available
reference is itself converged only to ~1e-3, so the `Gap_F` column has no better
resolution than that on those networks. A compressed row whose true gap is below the
reference's own suboptimality cannot be distinguished from zero.

### Decision B (2026-07-19) — repetitions, and large networks first

Timings repeat **3x on small instances and once on the large ones** (Sketch, Regional,
Philadelphia). `Gap_F`, `delta_F` and `R^2` are deterministic and unaffected by the repeat
count; only the CPU column loses its dispersion, and the `reps` column marks every row so
the reader can see it. **The large networks are run first**, since they set the campaign's
critical path.

## Evaluation contract (`python/v3_metrics.py`, Gate 0 = `test_v3_metrics.py`)

`x^F = P_X(x~)` per OD over `X = {x >= 0, Ax = d}`; then

    delta_F(%) = 100 * ||x^F - x~||_1 / max{1, ||x^F||_1}
    Gap_F(%)   = 100 * (f(v^F) - f(v^ref)) / f(v^ref)
    R^2        = 1 - SSE(v^F, v^ref) / SST(v^ref)

**Two findings from Gate 0 that constrain the campaign:**

- `delta_F` is **provably independent of the conversion map**. Per OD, both the Euclidean
  projection and clip-and-rescale move L1 mass `sum_{x<0}|x| + (sum_{x>=0} x - d)`
  (Euclidean: `k*theta = sum_S x - d`, clipped terms contribute `|x_j|`). Verified
  numerically to 1e-8. So the `P_X` ambiguity cannot corrupt the `delta_F` column.
- `Gap_F` is **strongly map-dependent**: on one test iterate, Euclidean gives `+12.62%`
  and clip-and-rescale `-0.63%`. Since the whole point of the v3 metric is that `Gap_F`
  be nonnegative, the campaign uses the **Euclidean projection** (which is what `P_X`
  denotes) for the paper, and records the rescale value in a `rescale_*` diagnostic
  column so the choice stays visible.
- `Gap_F >= 0` was confirmed against a properly converged optimum (projected-gradient
  norm 2.8e-14): 0/300 perturbed iterates negative, worst case exactly 0. A "best of N
  random feasible points" reference is *not* sufficient to test this and produced 37/300
  false negatives.

## Cell map

### Table 2 — `tab:operator-consistency` (23 cells) → `results/table2_consistency.csv`
One existing test instance, matched stopping tolerances. Columns: raw obj. diff. (%),
`||A x~ - d||_inf`, `delta_F` (%), feasible obj. gap (%), link-flow diff. (%).

| row | representation | method | driver |
|---|---|---|---|
| 1 | Full (tau=0) | ALM | `run_table2_consistency.py` |
| 2 | Full (tau=0) | Frank-Wolfe | " |
| 3 | Full (tau=0) | Gradient projection | " |
| 4 | Existing (tau,r) | ALM | " |
| 5 | Existing (tau,r) | FW or GP | " |

Rows 1's last two cells are pinned to 0.000 by definition (ALM full *is* the reference).

### Table 4 Panel A — `tab:cutoff-summary` (11 cells) → `results/table4A_thresholds.csv`
3 networks x 4 thresholds. Fills `delta_F` and `Gap_F`; per decision 2 the CPU, iteration
and `R^2` columns are also re-timed and re-emitted rather than carried over.
Thresholds preserved exactly as submitted: Sketch 0/0.46/1.06/4.54, Regional
0/0.23/0.46/1.03, Philadelphia 0/0.67/0.99/5.33. Reference = that network's tau=0 ALM run.

### Table 4 Panel B — `tab:cutoff-summary` panel B (28 cells) → `results/table4B_richness.csv`
One existing network, fixed OD set, `K` in {8,16,32,64}, `r = 50`. Columns: `q`
represented/OD, `K/q`, full dim `n`, compressed dim `s+r`, ALM speedup, GP speedup,
feasible gap (%). The GP column is produced twice (`GP-signed`, `GP-grouped`) per
decision 1; the manuscript cell takes `GP-signed`, which is v3's own representation.

### Table 5 — `tab:impact-rank` (8 cells) → `results/table5_rank.csv`
Regional (tau=1.03) and Philadelphia (tau=5.33), `r` in {50,100,150,200}. Fills
`Gap_F`; CPU/iterations/`R^2` re-timed per decision 2.

## Gates

- **Gate 0** — `test_v3_metrics.py` PASSED 2026-07-19 (7/7 checks).
- **Gate 1** — every `tau=0` reference reaches its stated tolerance and self-evaluates to
  `Gap_F = 0`, `R^2 = 1`.
- **Gate 2** — no `Gap_F` in any table is negative beyond solver tolerance. A negative
  value means the reference is under-converged, not that compression beat the full model.
- **Gate 3** — `S_R * S_C|R = S_RC` style identities where reported; `K/q` consistent with
  `n` and `s+r`.
- **Gate 4** — every rendered LaTeX row traces to a CSV row (checked by `fill_tables.py`).

## Timing protocol

Single-thread (`OMP_NUM_THREADS=1`), quiet machine, median of 3 independent process
invocations for anything reported as CPU. Preprocessing (nominal flows, SVD) is timed and
reported separately from the online solve, per the manuscript's own instruction.
