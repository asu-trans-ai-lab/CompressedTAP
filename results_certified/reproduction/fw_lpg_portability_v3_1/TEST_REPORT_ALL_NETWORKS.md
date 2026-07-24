# FW/Origin Extension — Full Test Report (all networks)

Package: compressed_TAP_repro_package_v1_1_FW_origin. Environment:
Linux, Python 3.12, NumPy 2.4.4, SciPy 1.17.1, g++ 13. Seed 7, quick
grid (K ∈ {8,16,32}; full-grid K=64 exceeds the per-call compute budget
here and is unrun; the quick grid matches the shipped references
exactly, see below).

## What was executed

1. Unit tests: `pytest tests/` — 3/3 passed.
2. Controlled route-rich experiments: FW (4 representations × 3 K) and
   origin-block (2 × 3 K).
3. Both real networks: FW (5 representations × 3 K each) and
   origin-block (K=8), Sioux Falls and Chicago Sketch, target v/c 2.0.
4. C++ atom kernel: built clean; run on both exported K=32 CSR cases.
5. `validate_extension.py results/quick_final`: **46/46 checks passed**
   (feasibility ≤ 1e-8, nonnegativity ≥ −1e-10, grouped speedup > 1 and
   monotone in K, weighted screening more accurate than unweighted at
   K = 16 and 32, origin-block checks, C++ kernel speedup > 1).
6. Reference comparison: every deterministic column (objective gaps,
   link errors) in all four shared CSVs is **bit-identical** to the
   shipped reference results — 86 columns compared, maximum absolute
   difference 0. The quick_final set additionally contains the
   origin-controlled and Chicago origin rows absent from the shipped
   references.
7. Formulation-equivalence extension: FW on the full path pool versus
   the SLSQP reduced-gradient solve on identical instances (n_od=3,
   K=16, v/c 2.0). Chicago Sketch: relative difference 4.1e-15
   (machine precision). Sioux Falls: 1.8e-5, FW above SLSQP, the
   expected O(1/k) conditional-gradient tail at rgap 1e-9, not a
   formulation discrepancy. The chain ALM ≈ simplex-PG ≈ RG now
   extends to ≈ FW.

## Headline numbers

Controlled route-rich (20 OD): grouped G4-M1 speedup 1.06 → 1.45 →
4.64 over K = 8/16/32 at gaps 0.19 → 0.11 → 0.045%; origin-block
grouped 1.20 → 1.57 → 6.74 at the same gaps. Weighted QR screening
beats unweighted where screening binds: 0.015% vs 0.31% (K=16),
0.020% vs 0.245% (K=32) — the Step 2A accuracy-per-dimension mechanism
survives inside FW as feasible path screening. Sioux Falls network:
every representation exact (rgap reaches 0 — vertex equilibria; the
anchor-supported regime again), grouped speedups 1.6–1.75×. C++ CSR
atom kernel: 3.23× (SF), 1.79× (CS) at K=32.

## Two flags the validator does not catch

1. **Matched-accuracy caveat on three Chicago network rows.** At v/c
   2.0 the Chicago rows FW-grouped-G3-M1 (K=16: gap 1.215%, "speedup"
   15×; K=32: gap 0.53%, 183×) and FW-unweighted-QR-screened (K=16:
   gap 1.216%, 85×; K=32: 0.53%, 151×) show rgap = 0 with large
   speedups: the restricted feasible set converges essentially
   instantly to its own, worse, optimum, while the full pool spends
   iterations reaching rgap 1e-5. These speedup entries are
   time-to-own-optimum, not matched-accuracy timings, and must be read
   jointly with their gap column — precisely the discipline of the
   revised protocol (Section 6.2). Recommended validator addition: at
   the network level, either condition the speedup check on
   gap < 0.25% or add a time-to-common-accuracy column. The
   controlled-grid checks are unaffected (grouped gaps there are
   0.045–0.19% with genuine matched-tolerance timing).

2. **The support-coverage mechanism reappears, and the extension
   confirms it.** Chicago at v/c 2.0 is the spread regime: 3 groups +
   1 major per OD is insufficient (gaps 0.25–1.22%), while 6 groups +
   2 majors is exact at every K with modest 1.02–1.11× speedup. This
   is the accuracy–dimension frontier predicted by the share-ablation
   and oracle results of the main package, now observed inside FW.
   The G6-M2 configuration, or G3-M1 plus rollout-inspired promotion,
   is the recommended Chicago setting; G3-M1 alone should not be
   quoted for Chicago at this congestion level.

## Verdict

The extension does what its README claims: grouped nonnegative atoms
are a valid FW/origin-block representation — the OD-simplex geometry,
the LMO, and the convex-combination iterate structure are preserved
(consistent with the convexity-preservation proposition), every
iterate is feasible to 1e-8, and the route-rich speedup mechanism
carries over to both algorithm families and to the C++ kernel. All
deterministic results reproduce the shipped references exactly. The
two flags above are reporting-discipline items, not correctness
defects, and both have one-line fixes.
