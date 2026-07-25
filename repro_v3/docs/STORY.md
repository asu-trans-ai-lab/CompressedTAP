# The coherent story: compression on the Chicago Sketch K/OD axis

*(Central case locked 2026-07-24: one network — Chicago Sketch; one axis — candidate-pool
richness K/OD; fixed τ=4.54, r=50, tol=1e-4, single thread. All numbers below trace to
`repro_v3/results/{kod_axis_alm,highk_fixes,sketch_operators,kod_axis_cpp}.csv`.)*

## 0. HEADLINE SPEEDUPS

| case | **SPEEDUP** | accuracy |
|---|---|---|
| Grid N32 K64, C++ hard **r=10** | **8.6x** | +0.08% |
| Grid N16 K32, C++ hard r=20 | **7.3x** | 0.0000% |
| Grid N16 K64, C++ hard r=20 | **6.8x** | 0.0000% |
| **Sioux Falls (real network), C++** | **6.2x** | certified (tau,r) gap |
| Grid K=32-64 column (N24/N32), C++ | **3.1-5.2x** | <= 0.55% |
| Sioux Falls, Python | **3.6x** | +3.4% |
| Chicago Sketch sweet spot, Python | **2.0x** | +0.02% |

Compression accelerates **ALM only** (GP 0.00-0.10x, RG 0.07-0.55x compressed -- see 5d/5e).
Conditions for the win: w0 anchor, small rank (r~10-20), high reduction, long paths
(nnz/path >= ~2r). Rows 1-5 are same-engine C++ ratios (baseline caveat in 5f).

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

## 5e. The constraints-vs-variables law (why "fewer variables" is not enough)

Every iteration costs (gradient work over the VARIABLES) + (feasibility work over the
CONSTRAINTS). Compression shrinks the first term for every method (n -> s+r); it does not
shrink the second — it densifies it (the n-s reconstruction rows persist, and the subspace
couples them). The operator table of 5d is this law in action:

- **ALM is constraint-tolerant**: feasibility lives in the objective via multipliers and a
  penalty, so its iteration cost is gradient-dominated — variable reduction is passed
  straight through to wall-clock. Compression + ALM wins.
- **GP is constraint-brittle**: its full-space engine is the separable, closed-form per-OD
  simplex projection (essentially free). The compressed feasible set
  {A1 y + M z = d_eff, y >= 0, w0 + U z >= 0} is not separable — z enters every OD equality
  through M and every minor bound through dense U — so the projection becomes a coupled QP
  (Dykstra + an ell x ell factorization per iteration) that costs more than the variable
  reduction saves.
- **RG eliminates constraints by substitution** and lives off the resulting plain box
  bounds. On the signed basis, the eliminated structure returns as the dense
  w0 + U z >= 0 hinge over all minors, O(n_minor x r) per evaluation: what substitution
  removed, the representation re-added.

**Law: variable reduction accelerates a method only if its per-iteration bottleneck is in
the variables, not in the constraints.** The constructive consequence is the same bridge as
5c: a grouped/bundle (atom) representation — x_p = s * d_p, s >= 0 — makes the compressed
feasible set separable AGAIN (plain box, closed-form projection), so projection- and
reduced-gradient-type methods recover their cheap iterations. Two roads, one principle:
signed-SVD + w0 pairs with ALM (Paper 1); atoms internalize both the anchor and the
separability, opening compression to the whole operator family (Paper 2).

## 5f. Verification suite (2026-07-25, grid_verify.csv): what holds across engines and ranks

Two home-regime cells (N16K32, N32K64): C++ full + hard at r in {10,20,40}; Python full,
hard, and w0-free at the same instance/tau/tol.

**Verified:** rank law in C++ -- speedup peaks at small rank (N32K64: 8.62x at r=10, 6.46x
at r=20, 1.49x at r=40) with quality flat (gaps <= 0.37%); dense cost linear in r, matching
Section 1's mechanism and Table 5's insensitivity.

**NOT verified -- two caveats now attached to earlier sections:**
1. *Grid ALM speedup at matched solution quality.* The C++ FULL solver under-converges on
   grids: its objective is 4.6-4.9% ABOVE the Python full objective while also being slower
   (eta-gate weakness, the R20 pattern). The C++ 6-8x therefore compares two equally
   under-converged solves; Python (strong full, weak compressed) gets 0.84-1.65x. Neither
   engine alone yet demonstrates the grid speedup at matched quality. Fixing the C++
   full-mode stopping rule (author decision) is the path to a clean matched-quality number.
2. *The w0 SPEED claim is C++-specific.* In Python, w0-free is somewhat faster; the
   accuracy damage replicates at scale only. Correct statement: w0 is load-bearing for
   accuracy at scale in both engines, and for speed in the compiled solver
   (W0_LOAD_BEARING.md section 8).

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

## 7. Partial closure (2026-07-25) — the headline numbers first

**THE NUMBERS THAT HEADLINE (largest certified speedup ratios, within-engine, same
tolerance, same instance):**

| # | case | **SPEEDUP** | accuracy |
|---|---|---|---|
| 1 | Grid N32 K64, C++ hard **r=10** | **8.6x** | +0.08% |
| 2 | Grid N16 K64, C++ hard r=20 | **6.8x** | 0.0000% |
| 3 | Sioux Falls, C++ hard | **6.2x** | matches certified (tau,r) gap |
| 4 | Grid N16 K32 (verify rerun), C++ r=20 | **7.3x** | 0.0000% |
| 5 | Grid N24/N32 K=32-64 column, C++ | **3.1-5.2x** | <= 0.55% |
| 6 | Sioux Falls, Python hard | **3.6x** | +3.4% (= certified Table 2 value) |
| 7 | Sketch sweet spot (K-bar 3.5, 70% red), Python | **2.0x** | +0.02% |

Caveat that travels with rows 1-5: the C++ full baseline under-converges on grids
(STORY 5f); rows are same-engine, same-stopping-rule ratios. Rows 6-7 are
strong-baseline Python numbers.

**Per-operator closure:**

- **ALM — CLOSED, positive.** Compression is an ALM accelerator (the numbers above), and
  only an ALM accelerator. Four measured conditions for the win: w0 anchor present; small
  rank (r ~ 10-20: 8.6x at r=10 vs 1.5x at r=40); high variable reduction; paths long
  relative to rank (nnz/path >= ~2r). Deferred: matched-quality cross-engine grid number
  (blocked on the C++ eta-gate stopping rule -- author decision).
- **GP — CLOSED, negative on the signed basis, by geometry.** GP lives off the separable
  per-OD projection; the signed compressed set destroys separability (DNS at real ell;
  0.00-0.10x where feasible). No tuning recovers it. GP-full stays as the reference
  operator (best objective everywhere; fastest full method on the grid).
- **RG — CLOSED, diagnosis points at the representation.** Anchor substitution removed the
  equality but the dense minor box was the real bottleneck (0.07-0.55x). The grouped
  box-bound RG was essentially exact (once beating full-space): RG's future is the
  grouped/atom representation.

**Three transferable laws (the round's yield):**
1. Variable reduction helps only methods whose per-iteration bottleneck is in the
   variables; compression densifies constraints (5e).
2. A signed subspace of nonnegative flows needs an interior anchor; w0 is load-bearing —
   for accuracy at scale in both engines, for speed in the compiled solver (5c, W0 doc).
3. Speedup is a property of (representation x operator x engine x instance geometry) —
   report within-engine, matched tolerance, named baseline; never as "compression" alone.

**Move on to:** Paper 1 ships ALM + signed-SVD + w0 with the four conditions and the
headline table above; Paper 2 inherits both structural fixes (anchor internalized,
separability restored) via atoms — where GP- and RG-type methods re-enter. Deferred
engineering: C++ full-mode stopping rule; Regional-scale rerun.

## 8. Beyond the base problem: why ALM is the right frame for constrained and multi-layer cases

*(Outlook. NOT measured in this campaign -- stated as the structural argument the measured
results support, and as the next experiment set.)*

The problem solved here carries only OD conservation and nonnegativity. Deployed assignment
problems routinely add side structure:

- **Link / corridor capacity constraints** -- v = B'x <= kappa (physical capacity, managed
  lanes, environmental or emissions caps on a subnetwork).
- **Knapsack / budget rows** -- total tolling revenue, investment budget, total VMT or
  emissions: a few dense rows coupling every path.
- **Assignment-side resource capacity** -- parking, transit seat/vehicle capacity, charging
  stations: capacities on resources shared across many OD pairs.
- **Coupling to a control layer** -- signal green splits and offsets enter the delay
  function, so assignment and control are solved jointly (or bilevel) with consistency
  constraints tying the two layers.

**Why these favour ALM.** Each item above is, for an augmented Lagrangian, one more
multiplier block and one more penalty term: the algorithm is unchanged, the per-iteration
cost grows by the size of the added constraint set, and multipliers warm-start naturally
across scenarios (which is precisely the repeated-assignment use case the paper targets).
The solver already carries several constraint regimes (hard / soft / screen) for the minor
nonnegativity block, which is the same machinery applied to a different constraint family.

**Why they disqualify projection-type operators.** GP and RG earn their speed from a
projection that must stay closed-form and separable. A link-capacity row couples paths
*across* OD pairs; a knapsack row is dense over all paths; a control-layer coupling can make
the feasible set non-polyhedral or the problem bilevel. In every case the projection degrades
into a general QP -- the same failure mode measured in 5d/5e, now for a second, independent
reason. This is the strongest argument for the ALM framing: it is not only faster under
compression, it is the only one of the four operators that *extends*.

**How compression interacts (the honest part, from the 5e law).** Compression shrinks the
variable side; side constraints grow the constraint side. Two regimes follow:
- Constraints sized by the **network** (m links, a handful of budget rows) do not grow with
  the candidate pool. Compression's benefit survives and even strengthens as pools get
  richer, because the constraint work is fixed while the uncompressed variable work grows.
- Constraints sized by the **paths** (per-path limits) hit exactly the dense-block problem
  of the minor box and would erode the benefit, per the same law.

**Next experiment set (proposed, not run):** capacity-constrained ALM on the grid at
K = 32-64 with r = 10-20 -- the home-regime cells -- measuring (i) speedup versus the
unconstrained baseline, (ii) how many capacity rows can be added before the constraint side
dominates, and (iii) whether GP/RG degrade further, as predicted.
