# The coherent story: compression on the Chicago Sketch K/OD axis

*(Central case locked 2026-07-24: one network — Chicago Sketch; one axis — candidate-pool
richness K/OD; fixed τ=4.54, r=50, tol=1e-4, single thread. All numbers below trace to
`repro_v3/results/{kod_axis_alm,highk_fixes,sketch_operators,kod_axis_cpp}.csv`.)*

## 1. The central curve (Python, signed-SVD ALM-hard vs full ALM)

| pool | K/OD | reduction | speedup | accuracy gap |
|---|---|---|---|---|
| V2 | 2.45 | 53.1% | 0.98× | +0.03% |
| E0 | 3.49 | 70.3% | **1.95×** | +0.02% |
| E2 | 8.64 | 87.1% | 0.66× | +0.59% |
| K10 | 11.09 | 89.8% | 0.67× | +1.12% |
| K15 | 15.34 | 92.4% | 0.67× | +1.86% |

**Finding 1 — a sweet spot, not a monotone law.** Speedup peaks at ~2× where the variable
reduction is already high (70%) but the minor set is still small in absolute terms. Beyond
the peak, the dense reconstruction w₀+U_r z ≥ 0 — evaluated on all n_minor rows, twice per
gradient call, cost O(n_minor·r) — outgrows the dimension saving, and the falling branch
settles at a flat 0.66–0.67× while the accuracy gap grows monotonically (+0.03% → +1.86%).
At high K/OD the signed-SVD representation loses on speed *and* accuracy simultaneously.

## 2. Why the original expectation ("richer pools ⇒ bigger wins") half-holds

The expectation is right about the *representation* (K/q compression factor does grow — the
compressed dimension stays ~s+r while n = ℓ·K̄ grows) and wrong about the *solver cost*: the
hard-regime feasibility check never shrinks below n_minor rows. The submitted paper's own
"structural limitation" paragraph states exactly this; the axis makes it quantitative and
locates the crossover.

## 3. Fixes tested on the falling branch (E2, K10)

| fix | E2 | K10 | verdict |
|---|---|---|---|
| rank r=10 | 1.18× (+0.62%) | 0.09× (+1.17%) | fragile: helps at E2, catastrophic at K10 (subspace too small → ALM thrashes) |
| rank r=25 | 0.90× | 0.98× | breakeven |
| grouped box-bound | 0.34× (+0.016%) | 0.54× (**+0.0000%**, beats full) | **accuracy champion**: feasibility by construction, all-sparse; slow only because the scipy L-BFGS implementation iterates heavily — the structural properties (O(nnz)/eval, no dense block) are exactly what the falling branch needs |

**Finding 2 — the representation, not the rank, is the lever.** Shrinking r trades the dense
cost against subspace quality and can collapse. The grouped (bundle) representation removes
the dense block entirely and is essentially exact — at K10 it found a *better* feasible point
than the full-space solve. Its current implementation does not win on wall-clock; a compiled
implementation is the natural next step (consistent with the earlier certified finding that
the latent-atom SVD fails its home regime and the bundle model is the answer).

## 4. Operator comparison at K=2.45 (matched certificate 1e-4)

| operator | space | time | note |
|---|---|---|---|
| FW | full | **1.4 s** | dominates at working tolerance (sublinear tail only hurts at tight tol) |
| GP | full | 21.1 s | best objective |
| ALM | full | 21.6 s | |
| ALM-hard | compressed | 23.2 s (0.93×) | |
| RG-anchor | compressed | 38.1 s (0.55×) | conservation exact by construction |
| GP-signed | compressed | DNS | exact projection needs dense ℓ×ℓ factorization — architectural, ℓ=17,464 |

**Finding 3 — the honest baseline matters.** At loose/working tolerance, plain Frank–Wolfe is
the strongest full-space method on lean pools; compression arguments must be made against it,
not only against ALM-full.

## 5. Engine dependence (C++, tuned -O3 build; bit-identical objectives)

The C++ axis (kod_axis_cpp.csv) shows the speedup is a property of the ENGINE's relative
dense/sparse throughput, not of the representation alone:

| pool | K/OD | Python speedup | C++ speedup | C++ gap |
|---|---|---|---|---|
| V2 | 2.45 | 0.98× | 0.34× | +0.000% |
| E0 | 3.49 | **1.95×** | 0.78× | +0.002% |
| E2 | 8.64 | 0.66× | 0.81× | +0.000% |
| K10 | 11.09 | 0.67× | 0.75× | +0.122% |
| K15 | 15.34 | 0.67× | 0.99× | +0.196% |

Opposite shapes: numpy (BLAS-fast dense gemv, slow sparse ops) peaks at the sweet spot and
falls; compiled C++ (fast sparse loops, SSE2-bound dense gemv) rises monotonically toward
breakeven but never crosses 1 on Sketch. Compression's only outright C++ win is Sioux
(full 7.6 s vs hard 1.6 s = **4.75×**; Python 3.63×), where the win is iteration-count-driven
(1,246 vs 4,988 inner) rather than per-iteration. This reproduces the certified 2026-07-16
finding that the compression speed ratio is engine-dependent (then measured as 41× vs 0.4×).
Build note: -O3/-march=native and ANY AVX2 codegen segfault full mode (MinGW alignment);
-O3 baseline-SSE2 is the fair single binary (bit-identical objectives; full 1.4× / hard 1.6×
faster than -O2).

## 5b. The grid mechanism case (2026-07-25, ALL C++): the home regime found

N x N Manhattan grid, left-column -> right-column demand (q=720/N so mid-cut v/c ~ 1.2 at
every N), C++ ksp_gen pools of depth K, logit x0, tau = 75th pct of x0 (~74% reduction),
r=20, o3sse build, full-vs-hard within-engine (grid_axis_cpp.csv):

| speedup | K=4 | K=8 | K=16 | K=32 | K=64 |
|---|---|---|---|---|---|
| N=8  | 0.55 | 0.59 | 1.53 | 1.45 | 2.01 |
| N=16 | 2.09 | 0.89 | 1.75 | 5.16 | **6.82** |
| N=24 | 2.25 | 1.18 | 0.73 | 3.87 | 4.86 |
| N=32 | 1.76 | 2.76 | 0.96 | 3.13 | 5.02 |

- **On the grid the original claim holds: speedup RISES with column-generation depth K for
  every grid size**, reaching 6.8x at gap 0.0000% (N=16 K=64); the K=32/64 columns are
  uniformly 3.1--6.8x with gaps <= 0.55%.
- **Mechanism law (reconciles grid vs Sketch):** the compressed gradient costs ~2r dense
  flops per minor row; the full gradient costs ~nnz-per-path sparse flops per path. Grid
  paths are long (nnz/path ~ 35-45 at N=32 >= 2r=40) -> compression wins and keeps winning
  as K grows. Sketch paths are short (nnz/path ~ 15 << 2r=100 at r=50) -> the dense block
  loses. Compression's home regime = long-path, many-parallel-route networks with rich
  candidate pools; τ should keep reduction high and r small relative to path length.
- Footnotes: low-K cells carry 1.4--3.4% gaps (75th-pct τ cuts real flow when the pool is
  thin); the K=8-16 mid-column is noisy on sub-10 s solves.

## 5c. The w0 ablation (2026-07-25, ALL C++): the offset is load-bearing

Same grid matrix, identical split/U/D/M, only the affine offset removed (w0=0, with its
flow moved consistently out of v_base and back into d_eff). Result (grid_w0free_cpp.csv):
mean speedup drops 2.60x -> 1.07x, mean gap grows 0.75% -> 2.42%; the best cells collapse
(N=24 K=64: 4.86x -> 0.45x; N=16 K=64: 6.82x -> 1.86x; N=32 K=32: 3.13x -> 0.32x, worst gap
6.1%). Mechanism: without the offset the box U z >= 0 with signed U pins z ~ 0, so the
demand w0 used to pre-load must be fought back through the feasibility constraint.
**Verdict: keep w0 — it is the mechanism, not an add-on.** (Closes the author's long-open
"speedup ratio related to w0" question; per decision 3 the variant stays a CSV diagnostic.)

## 5d. Operator x compression on the grid (2026-07-25): the benefit is ALM-specific

Six home-regime cells (N in {16,32} x K in {16,32,64}), same w0 representation, Python
(the one engine with all four operators), within-family speedups (grid_operators.csv):

| cell | ALM | GP-signed | RG-anchor | FW-full baseline |
|---|---|---|---|---|
| N16K16 | 1.11x | 0.00x | 0.07x | 5.0 s |
| N16K64 | 1.96x | 0.01x | 0.13x | 17.4 s |
| N32K16 | 3.85x | 0.02x | 0.34x | 31.6 s |
| N32K64 | 2.54x | 0.10x | 0.26x | 61.1 s |

- **Only ALM benefits from compression.** Structural reason: ALM's inner subproblem needs
  only gradients in (y,z), so the subspace shrinks its work directly. GP and RG live off the
  cheap separable per-OD projection of the full space; compression destroys that
  separability (equality couples y and z through Mz, the box through dense U), so their
  compressed projections cost more than what they replaced. GP-signed caps out everywhere;
  RG-anchor runs 2-8x slower than ALM-hard at the same terminal accuracy.
- On the grid, GP-full is the strongest full-space method (best objective, cert 1e-4,
  0.6-30 s); FW loses its Sketch advantage here.
- Honesty caveat: the Python compressed rows terminate at cert ~5e-2 vs ~1e-4 for the full
  rows, so this table's cross-space ALM ratios (1.1-3.9x) mix speed with looseness; the
  matched-accuracy ALM speedup is the C++ matrix of Section 5b (up to 6.8x, gaps <= 0.55%).
  The clean claim from THIS table is the operator ranking under compression.
- **Paper consequence: the ALM framing is not a preference, it is the measured pairing** --
  compression + ALM works; compression + projection methods does not.

## 6. What the paper should say (headline speedups)

1. Compression delivers **~2× at its sweet spot** (moderate richness, ~70% reduction) with
   negligible accuracy loss (+0.02%), and up to **3.6–4.8×** on small instances (Sioux 91%
   reduction, both engines) — but the sweet spot's location and height are engine-dependent
   (BLAS-backed dense vs compiled sparse), and this must be stated with the claim.
2. The benefit is **conditional, with a characterized crossover**: below ~40–50% reduction
   nothing is gained; far beyond the sweet spot the dense minor block dominates and the
   signed-SVD representation should not be used for speed.
3. τ and r form a **speed–accuracy dial** with a fragility boundary (too-small r can
   destabilize the inner solver).
4. For high-richness regimes the **grouped box-bound representation** is the right object:
   essentially exact, structurally free of the dense block; wall-clock competitiveness is an
   implementation (compiled) question, stated as future work.
5. All speedup claims are within-campaign ratios at matched tolerance, single thread, with
   the strongest applicable full-space baseline named (FW at working tolerance).
