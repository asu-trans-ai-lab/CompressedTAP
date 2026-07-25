# The affine offset w0 is load-bearing

*Standalone note, 2026-07-25. All numbers C++ (`compressed_solver_o3sse.exe`, -O3 baseline-SSE2,
bit-identical objectives to the certified -O2 build), single thread, matched settings. Data:
`results/grid_w0free_cpp.csv`, `results/grid_axis_cpp.csv`, the Sioux runs below; ablation
fixer: `drivers/run_grid_w0free.py`.*

## 1. The claim

In the compressed representation of minor path flows

    x_minor = w0 + U_r z,        feasibility  w0 + U_r z >= 0,

the affine offset w0 (the nominal minor flow) is not a convenience feature. It is the
mechanism that makes the signed-SVD subspace usable: it places the feasible box's interior
at the starting point. Removing it (w0 = 0, with its flow moved consistently into the
demand terms so the model stays exact) degrades the method decisively on both a controlled
grid family and the Sioux Falls real network.

## 2. The ablation is exact, not a re-tuning

Identical instance, split, U, D, M. Only three consistent blob edits:
v_base <- v_base - B2'w0 (the nominal minor flow is no longer pre-loaded on links),
d_eff <- d_eff + A2 w0 (its demand returns to the explicit variables), w0 <- 0.
Nothing else changes; the same solver runs on both.

## 3. Grid evidence (mechanism testbed: N x N, K-shortest pools, 20 cells)

| aggregate over the 20 (N,K) cells | with w0 | w0-free |
|---|---|---|
| mean speedup vs full | **2.60x** | 1.07x |
| best cell | **6.82x** (N=16, K=64, gap 0.0000%) | 3.33x |
| worst cell | 0.55x | **0.32x** |
| mean feasible-objective gap | 0.75% | **2.42%** |
| worst gap | 3.37% | **6.07%** |

The collapse concentrates exactly where compression is strongest: N=24 K=64 falls
4.86x -> 0.45x, N=16 K=64 falls 6.82x -> 1.86x, N=32 K=32 falls 3.13x -> 0.32x with the
gap growing to 4.6%. Only two sub-noise exceptions in 20 cells.

## 4. Sioux Falls evidence (real network, tau=600, r=50, median of 3)

| | with w0 | w0-free |
|---|---|---|
| compressed-hard time | **1.7 s** | 7.9 s |
| speedup vs full (10.6 s) | **6.2x** | 1.34x |
| feasible objective | 4,378,934 | 4,378,119 (~same) |
| inner iterations | 1,246 | 1,420 |
| cost per inner iteration | **1.4 ms** | 5.6 ms |

On the real network the damage is almost purely speed: per-iteration cost quadruples at
nearly unchanged iteration count and accuracy.

## 5. Mechanism

With w0 > 0 the starting point z = 0 sits in the interior of the box {w0 + U_r z >= 0}:
most minor rows are comfortably feasible, the hinge penalty is inactive almost everywhere,
and the subspace is free to do its work. With w0 = 0 the box becomes {U_r z >= 0} with a
signed basis U_r: z = 0 lies on the boundary of every row at once. Two consequences,
each visible in the data:

- **Speed** (dominant on Sioux): the hinge is active exactly where the solver iterates, the
  ALM landscape is nonsmooth at the iterate, line search backtracks heavily, and multipliers
  churn — per-iteration cost x4 at similar iteration counts.
- **Accuracy** (dominant on the grid at scale): the constraint pins z ~ 0, the represented
  subspace is effectively lost, and the compressed solution degrades toward a majors-only
  solution — gaps grow up to 6%.

In one sentence: **a signed subspace of nonnegative flows is only usable relative to an
interior anchor; w0 is that anchor.**

## 6. The bridge to Paper 2: this is the road to latent atoms

The same principle, read forward, is the design argument for the latent-atom / bundle
approach of Paper 2:

- The signed-SVD span needs its anchor supplied EXTERNALLY (w0). Where the anchor is good,
  the method wins (grid, up to 6.8x; Sioux 6.2x). Where the offset must carry too much
  structure, its absence — or its misfit — is the dominant error source (the certified
  2026-07-19 ablation identified the offset as the dominant error source in support-sparse
  regimes; this note shows its removal is worse still).
- A **latent atom / bundle internalizes the anchor**: an atom is itself a nonnegative flow
  shape, and the decision variables are nonnegative combination weights, so feasibility is
  a plain box bound by construction — no signed reconstruction, no external offset, no
  dense hinge. The grouped box-bound representation measured in this campaign is exactly
  that object, and it was essentially exact where the signed span struggled (gaps +0.016%
  and +0.0000% on the Chicago Sketch falling branch, in one case beating the full-space
  solve).
- The progression is therefore: signed span + anchor (Paper 1, fast in its home regime,
  fragile where the anchor misfits) -> anchored nonnegative atoms (Paper 2), where the
  load-bearing role of w0 is absorbed into the representation itself. This also matches
  the certified 2026-07-16 finding that the latent-atom signed SVD fails its home regime
  and "the bundle model is the answer."

## 7. Reproduction

    # grid matrix with w0 (writes grid_axis_cpp.csv)
    python run_grid_axis_cpp.py --Ns 8,16,24,32 --Ks 4,8,16,32,64
    # ablation on the cached exports (writes grid_w0free_cpp.csv)
    python run_grid_w0free.py
    # Sioux: make_w0free() on the tau600 export, then compressed_solver_o3sse hard, 3 reps
