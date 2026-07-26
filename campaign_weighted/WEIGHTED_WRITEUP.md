# How to say the flow-weighted result

The point stands on its own: it is a one-line change to the representation, it costs nothing
to compute, and it improves both axes. Below: a contribution bullet, an abstract sentence, and
a self-contained subsection that needs no prior experimental setup beyond naming the instance.

---

## 1. Contribution bullet (introduction)

> **Which subspace to keep.** The compressed representation fixes a rank-$r$ subspace for the
> minor path flows, and the natural choice --- the leading singular subspace of the minor
> path-link matrix $B_2$ --- treats a path carrying a thousand vehicles and a path carrying one
> as equally worth representing. We instead retain the leading subspace of
> $\operatorname{diag}(\sqrt{w_0})\,B_2$, which minimises the reconstruction error *weighted by
> the nominal flow*. The change costs nothing (the two factorisations take the same time) and
> improves both the solution time and the accuracy, with the gain increasing in the richness of
> the candidate path set.

## 2. Abstract sentence

> Choosing the retained subspace by flow-weighted rather than plain reconstruction error
> reduces the feasible objective gap by up to $70\%$ at no additional preprocessing cost.

## 3. Self-contained subsection

### The choice of retained subspace

The compression fixes a rank-$r$ subspace through $\Phi_r = U_r$, and the obvious choice is the
leading left-singular subspace of $B_2$. That choice minimises $\|B_2 - \Pi B_2\|_F$, in which
every minor path counts equally. It is the wrong objective here: what the solution needs is an
accurate *link load*, and a minor path contributes to the link load in proportion to the flow
it carries. A path with negligible flow can be reconstructed badly at no cost, while a
high-flow path cannot.

Weighting by the nominal flow makes the criterion match the use. Taking the singular value
decomposition of $\operatorname{diag}(\sqrt{w_0})\,B_2$ retains the subspace that minimises
$\sum_p w_{0,p}\,\|(B_2 - \Pi B_2)_p\|^2$, so the basis is spent where the flow is. The
modification is one line, and because the two decompositions have identical cost --- $0.94$s
against $0.97$s on Chicago Sketch at $r=20$ --- any difference in solution time is a
convergence effect rather than overhead.

Table X isolates the effect on Chicago Sketch, holding the network, the demand, the threshold
and the rank fixed and varying only the richness of the candidate pool.

| $\bar K$ | speedup (plain $\to$ weighted) | feasible gap (plain $\to$ weighted) |
|---|---|---|
| 2.45  | $1.55\times \to 1.62\times$ | $0.029\% \to 0.028\%$ |
| 3.49  | $3.34\times \to 3.71\times$ | $0.023\% \to 0.020\%$ |
| 8.64  | $1.34\times \to 1.55\times$ | $0.702\% \to 0.209\%$ |
| 11.09 | $0.66\times \to 0.83\times$ | $1.191\% \to 0.435\%$ |

The weighted basis is better on both axes at every point, and the margin grows with richness:
negligible at $\bar K = 2.45$, but a $26\%$ reduction in solution time and a $64\%$ reduction in
the objective gap at $\bar K = 11.09$. The mechanism is visible in the trend. A rich candidate
pool is rich precisely in paths that carry almost no flow; the unweighted criterion spends its
rank budget representing them, and the weighted criterion does not. This also explains why the
regime that was previously the least favourable --- rich pools, where the compressed problem
had both the largest gap and the smallest speed advantage --- is the regime the weighting
repairs most.

The effect is not uniformly free. On Sioux Falls, where the threshold removes $91\%$ of the
variables and the compressed gap is an order of magnitude larger, the weighted basis reduces
the gap by $11\%$ to $20\%$ relative but costs a factor of two to three in solution time: with
that much of the problem compressed, a basis aligned to the flow is harder to optimise over.
The weighting should therefore be read as making the subspace choice consistent with the
objective, which is a clear gain in the moderate-compression regime and a trade in the extreme
one.

---

## 4. What this lets you avoid saying

The result needs no threshold sweep, no rank study and no engine comparison to state: one
instance family, one varied quantity, two bases. That is why it can lead the experimental
section instead of following three pages of setup --- the reader learns the mechanism first
and reads the parameter studies afterwards as consequences.

## 5. Numbers to keep straight

- Preprocessing: $0.94$s weighted, $0.97$s unweighted (Sketch-E0, $r=20$) --- the "free" claim.
- Largest accuracy gain: $0.702\% \to 0.209\%$ at $\bar K = 8.64$ ($-70\%$).
- Largest speed gain: $0.66\times \to 0.83\times$ at $\bar K = 11.09$ ($+26\%$).
- Sioux trade: gap $-11\%$ to $-20\%$ relative, speed $\times 0.28$ to $\times 0.62$.
- Source: `campaign_weighted/results/exp01_rank.csv`, `exp02_richness.csv`.
