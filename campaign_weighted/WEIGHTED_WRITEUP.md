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

## 2. The result that carries the point (Chicago Sketch, route richness)

Network, demand, threshold and rank fixed; only the candidate pool varies.

| $\bar K$ | weighted | unweighted | speed | gap |
|---|---|---|---|---|
| 2.45  | $1.62\times$ / $0.0284\%$ | $1.55\times$ / $0.0291\%$ | $\times1.05$ | $-2.4\%$ |
| 3.49  | $3.71\times$ / $0.0202\%$ | $3.34\times$ / $0.0232\%$ | $\times1.11$ | $-12.9\%$ |
| 8.64  | $1.55\times$ / $0.2086\%$ | $1.34\times$ / $0.7024\%$ | $\times1.16$ | $\mathbf{-70.3\%}$ |
| 11.09 | $0.83\times$ / $0.4350\%$ | $0.66\times$ / $1.1914\%$ | $\times1.26$ | $\mathbf{-63.5\%}$ |
| 15.34 | $1.20\times$ / $0.8649\%$ | $1.15\times$ / $1.8651\%$ | $\times1.04$ | $\mathbf{-53.6\%}$ |

Better on **both** axes at **all five** points, and the accuracy gain exceeds $50\%$ at every
$\bar K \ge 8.64$. Mechanism: a rich pool is rich precisely in near-zero-flow paths; the
unweighted criterion spends rank representing them, the weighted one does not. This is why the
regime that was previously least favourable is the one the weighting repairs most.

## 3. The grid family qualifies it — state this too

Twenty $(N,K)$ cells, C++, $r=20$:

| aggregate over 20 cells | weighted | unweighted |
|---|---|---|
| mean speedup | $2.47\times$ | $2.58\times$ |
| best cell | $\mathbf{7.99\times}$ | $7.73\times$ |
| mean feasible gap | $\mathbf{1.112\%}$ | $1.285\%$ |

On the grid the weighting **buys accuracy and costs a little speed on average**, the reverse of
Sketch. The pattern within the matrix is what matters: at high richness it wins outright
($N{=}16$, $K{=}64$: $7.99\times$ / $2.33\%$ against $7.73\times$ / $3.75\%$; $N{=}32$,
$K{=}64$: $5.41\times$ / $\mathbf{0.000\%}$ against $3.87\times$ / $0.406\%$), while at
moderate $K$ it can lose speed for accuracy ($N{=}32$, $K{=}32$: $1.66\times$ / $0.029\%$
against $2.59\times$ / $0.446\%$). Several cells reach a gap of exactly $0.000\%$ under
weighting where the plain basis leaves $0.1$–$0.4\%$.

So the honest headline is **not** "weighting is uniformly better." It is:

> Weighting makes the subspace criterion consistent with the objective. Where the pool is rich
> — the regime the method is aimed at — it improves both speed and accuracy. Where compression
> is extreme, it converts part of the speed advantage into accuracy.

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
