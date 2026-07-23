# Does the affine offset w^0 improve performance?  (ablation, 2026-07-19)

Signed SVD is parameterized affinely, x_N = w^0 + U_r z with w^0 the nominal minor flow and
feasibility w^0 + U_r z >= 0.  This ablation toggles ONLY w^0 (AFFINE vs LINEAR/offset-free),
holding basis, rank, solver (SLSQP + analytic gradient), constraints, tolerances and the
R0-full reference fixed.  Harness fidelity check: it reproduces the frozen grid's published
weighted-vs-unweighted improvement factor exactly (I_f median 2.41 over 32 cells).

| regime | cells | AFFINE better | LINEAR better | AFFINE gaps | LINEAR gaps |
|---|---|---|---|---|---|
| synthetic frozen grid | 64 | 0 | **64** | 0.46 – 14 % | ~1e-13 (48 cells) / 0.96–1.7 % (16 cells) |
| real subproblems, v/c 1.05 (SF, CS; K=8/16/32; r=2,5) | 24 | 0 | **24** | 2.9 – 190 % | **1e-8 – 1e-11 (numerically exact)** |
| real subproblems, demand x4 (spread support) | 24 | 7 | 17 | 0.44 – 45 % | ~0 – 47 % |
| ... restricted to cells whose minor block carries flow | 16 | 7 | 9 | — | — |

## Findings

1. **w^0 does not help; in the support-sparse regime it is the dominant error source.**
   Every calibrated real subproblem has minor support exactly 0 (anchor/major-supported
   equilibrium).  There the offset injects nominal mass that a rank-r span cannot retract
   feasibly, producing the reported 100%+ signed-SVD gaps.  Removing it makes the SAME basis
   at the SAME rank numerically exact in all 24 cells.
2. **Only in the spread-support regime is w^0 defensible**, and there it is a coin flip
   (7/16 vs 9/16).  Where AFFINE does win it is the flow-weighted basis that wins
   (e.g. SF K=32 r=5: 17.1 % affine-W vs 19.6 % linear-W), i.e. the offset is worth keeping
   only when the optimum genuinely uses nominal mass.
3. **Implication for the weighted-vs-unweighted claim.**  I_f = eps_U/eps_W is measured in the
   AFFINE parameterization.  Removing w^0 collapses it to median 1.00 on the synthetic grid
   and to a tie at machine zero on the real subproblems.  Much of the measured benefit of flow
   weighting in those cells is therefore compensation for the offset rather than intrinsic
   basis quality.  In the spread regime, where the offset is appropriate, flow weighting keeps
   a genuine advantage.
4. **Sharper statement of the support-geometry principle.**  The paper attributes grouped-atom
   exactness to its span containing the anchor point, and signed-SVD failure to the affine
   offset.  This ablation confirms the diagnosis and completes it: the operative distinction is
   AFFINE vs CONIC parameterization, not signed vs grouped.  The grouped decoder is offset-free
   by construction; applying the same change to the signed basis recovers exactness.

## Caveat (do not overclaim)

In the support-sparse cells the optimum IS the zero vector on the minor block, which is exactly
the point the offset-free form represents perfectly at z = 0, so LINEAR's exactness there is
partly structural.  The independent evidence that it is not merely trivial: the 16 synthetic
cells whose minor block carries 31-36 % of demand also favour LINEAR 16/16.
