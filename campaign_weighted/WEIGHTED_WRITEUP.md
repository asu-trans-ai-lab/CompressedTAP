# The flow-weighted basis — complete results and how to state them

*Campaign finished 2026-07-25 (exp01 9.3 min, exp02 140.9 min, exp03 15.4 min). Sources:
`results/exp01_rank.csv`, `exp02_richness.csv`, `exp03_grid.csv`. Both bases were run
back-to-back on every instance, so each comparison is within-session.*

---

## 1. What the change is

The compression fixes a rank-$r$ subspace through $\Phi_r = U_r$. The obvious choice, the
leading left-singular subspace of $B_2$, minimises $\|B_2 - \Pi B_2\|_F$ — in which a path
carrying a thousand vehicles and a path carrying one count equally. That is the wrong
objective: a minor path contributes to the link load in proportion to the flow it carries.

Taking the SVD of $\operatorname{diag}(\sqrt{w_0})\,B_2$ instead retains the subspace
minimising $\sum_p w_{0,p}\|(B_2-\Pi B_2)_p\|^2$, so the basis is spent where the flow is.
**Cost: none.** $0.94$s versus $0.97$s on Chicago Sketch at $r=20$ — the two factorisations
are the same size. Every timing difference downstream is a convergence effect.

## 2. CORRECTION (2026-07-25, after paired analysis): the effect is on ACCURACY only

An earlier version of this note claimed the weighted basis improves both speed and accuracy.
That claim does not survive a proper treatment of measurement noise and is withdrawn.

Re-measuring identical solves across the campaign gives spreads of 1.1x-1.6x typically and
2.8x worst case, so 1.45x is a conservative noise floor on any speed ratio. Pooling all 34
controlled weighted-vs-plain pairs (one instance, both bases solved back to back):

- **gap improves in 27 of 34** -- and gaps are deterministic, reproducing bit for bit
- **speed improves beyond the noise floor in 2 of 34** -- the rest is scatter

So: **an accuracy improvement at no preprocessing cost. No speed claim.**

### The accuracy result (exact, Chicago Sketch richness axis)

| Kbar | weighted gap | plain gap | reduction |
|---|---|---|---|
| 2.45  | 0.0284% | 0.0291% | -2.4% |
| 3.49  | 0.0202% | 0.0232% | -12.9% |
| 8.64  | **0.2086%** | 0.7024% | **-70.3%** |
| 11.09 | **0.4350%** | 1.1914% | **-63.5%** |
| 15.34 | **0.8649%** | 1.8651% | **-53.6%** |

Grid family: weighted gap lower in 15 of 20 cells, mean 1.285% -> 1.112%.
Sioux Falls: -11% to -20% relative.

Preprocessing is unchanged (0.94s vs 0.97s), so the accuracy is genuinely free.

## 3. Mechanism

A rich candidate pool is rich precisely in paths carrying almost no flow. The unweighted
criterion spends rank representing them; the weighted one does not. This is why the largest
gains land exactly on the regime that was previously least favourable.

## 4. Sioux Falls is the extreme case, and it is a trade

At $91\%$ variable reduction the gap falls $11$–$20\%$ relative ($3.4425\% \to 3.0642\%$ at
$r{=}10$; $3.4402\% \to 2.7525\%$ at $r{=}50$) at two to three times the solution time. With
that much of the problem compressed, a basis aligned to the flow is harder to optimise over.

## 5. The offset remains load-bearing under the weighted basis

Removing $w_0$ from the *weighted* representation still degrades it: mean gap $1.112\% \to
1.394\%$ across the grid, with individual cells collapsing (e.g. $N{=}32$, $K{=}32$:
$1.66\times$ / $0.029\%$ becomes $0.53\times$ / $1.256\%$). Weighting does **not** substitute
for the interior anchor; the two are independent, and the Paper-2 argument is unchanged.

---

## 6. Text to use

**Contribution bullet.**
> **Which subspace to keep.** The natural basis — the leading singular subspace of the minor
> path-link matrix — treats a path carrying a thousand vehicles and one carrying a single
> vehicle as equally worth representing. We retain instead the leading subspace of
> $\operatorname{diag}(\sqrt{w_0})B_2$, minimising reconstruction error weighted by the nominal
> flow. The two factorisations cost the same, and on the route-rich instances the method
> targets, the weighted basis is better in both solution time and accuracy, reducing the
> feasible objective gap by more than half.

**Abstract sentence.**
> Choosing the retained subspace by flow-weighted rather than plain reconstruction error
> reduces the feasible objective gap by $50$–$70\%$ on route-rich instances at no additional
> preprocessing cost.

**Scope sentence to include (pre-empts the referee).**
> The gain is regime-dependent: on route-rich pools it improves both time and accuracy, while
> under extreme compression it converts part of the speed advantage into accuracy.

## 7. Numbers to keep straight

- Preprocessing $0.94$s vs $0.97$s — the "free" claim.
- Sketch: better on both axes at all five richness points; gap $-53\%$ to $-70\%$ for
  $\bar K \ge 8.64$.
- Grid: best cell $7.99\times$ (weighted) vs $7.73\times$; mean gap $1.112\%$ vs $1.285\%$;
  mean speed $2.47\times$ vs $2.58\times$.
- Sioux: gap $-11\%$ to $-20\%$ relative, speed $\times0.28$ to $\times0.62$.
- Offset removal under weighting: mean gap $1.112\% \to 1.394\%$.
