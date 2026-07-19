# Solver Portability: Latent Projected Gradient (LPG)

Question tested: is the latent compression gain a representation-layer
property, or a Frank--Wolfe artifact? Method: add exactly one thing --- a
second update operator --- and change nothing else. LPG (Latent Projected
Gradient on OD-demand simplices; deliberately not named reduced gradient:
no basis, no elimination, no reduced costs) shares with latent FW the
atoms A, the latent gradient A t(A'y), the OD demand simplices, the
objective, and even the stopping certificate (the identical FW duality
gap, computed from the same LMO). The update is Euclidean simplex
projection with a safeguarded BB trial step and Armijo backtracking on
the projected direction. Implementation: src/latent_projected_gradient.py
(~90 lines on top of the existing representation layer, as predicted).

## Protocol

Matched accuracy per representation: F*_rho computed to FW-gap 1e-10
(LPG primary, FW cross-check); each timed run stops at first crossing of
F*_rho (1 + 1e-6). Speedups are within-solver ratios
T(E-solver)/T(RC-solver); representation error (F*_rho - F*_E)/F*_E is
reported separately, never folded into a speedup. Timing repeats = 1,
single thread; deterministic columns are exact, times are indicative.

## Results (results/solver_portability/)

Controlled sweep, n_od=4, latent m=8 atoms/OD, K = 8 / 32 / 128 / 512:

| solver | E/RC matched-accuracy speedup over K |
|---|---|
| PG | 1.02, 1.17, 3.00, **7.70** |
| FW | 1.00, 1.07, 0.90, 2.46 |

Representation gaps (RC): 0.00, 1.19, 1.78, 1.85%; (C, no major):
0.00, 2.89, 6.26, 6.23%. Real networks at K=32, v/c 2.0: Sioux Falls
all-zero gaps and ~unit speedups (anchor-supported regime, both solvers
converge essentially immediately); Chicago Sketch RC gap 0.53% with
E/RC speedups 95x (FW) and 4.3x (PG); Chicago C gap **149.8%**.

## Four conclusions

1. **Representation gain != FW-specific gain.** The PG speedup grows
   monotonically with K/m, reaching 7.7x at K=512, exactly the
   dimension-dependence the portability argument requires: PG's
   per-iteration cost (loading + O(K log K) projection) scales with the
   representation, its iteration count does not. The prediction that
   compression benefits projection-based methods at least as much as
   linear-oracle methods is confirmed.
2. **The FW wall-time caveat is real and belongs in the paper.** In this
   Python implementation FW's time is iteration-count bound (the 1/k
   tail chasing a 1e-6 target) with size-independent per-iteration
   overhead, so its speedup stays near 1 until K=512. FW's
   per-iteration dimension gains are established separately at kernel
   level (C++ CSR 3.2x/1.8x; dense kernels up to 8.5x). Wall-time and
   kernel evidence must not be conflated.
3. **The support mechanism reappears, at its most dramatic.** Pure
   latent C (convex atoms with nominal shares, no explicit major)
   fails by 149.8% on the congested Chicago instance: nominal-share
   convex combinations cannot concentrate mass on the dominant path.
   One explicit major per OD (RC) repairs this to 0.53%. Same finding
   as the share ablation and the oracle bound, now inside the FW/PG
   geometry.
4. **LPG completes the equivalence chain and exceeds FW in depth.** In
   the five-solver integrated check (interface/integrated_check.py),
   LPG on the interface atoms matches SLSQP to 3.1e-14 on the spread
   instance where vanilla FW's tail stops at 9.8e-5 --- on identical
   atoms. The verified chain now reads
   ALM = simplex-PG = reduced-gradient = FW = LPG, all through the
   standard interface, gap-certified.

Recommended paper matrix, per the design note: main results E/C/R/RC
under FW; solver-robustness E/C/RC under PG; R stays a representation
baseline and is never called a reduced-gradient *algorithm*.
