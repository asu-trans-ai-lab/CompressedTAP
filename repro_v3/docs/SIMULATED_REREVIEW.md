# Simulated re-review of the current draft

*Grounded in the actual OPRE reports (`papers/review_comments.txt`, OPRE-2026-04-2974,
rejected 08-Jul-2026): Referee 1, Referee 2, and the AE. Each original objection is restated,
then answered as that reviewer would plausibly answer it after reading the current draft.
Written adversarially on purpose. Our own position is stated separately at the end.*

---

## Referee 1 — likely re-review

**On (1), limited computational advantage — his "most important concern."**

*Partly satisfied, and satisfied in a way that vindicates him.* His structural note predicted
the cause exactly: the nonnegativity block stays $(n-s)\times r$ and dense, so the per-iteration
gain is capped. The new draft adopts that diagnosis as its organizing principle
(\S cost model), states it as $2r$ versus links-per-path, and shows the consequences. A
reviewer rarely sees his own mechanism turned into the paper's spine; this is the single
strongest move in the revision.

*But he will not accept the headline.* He asked for an order-of-magnitude improvement. The
draft's largest numbers are $10.31\times$ on **Sioux Falls** (76 links) and $6.8$--$8.6\times$
on a **synthetic grid**. On the real networks the peak is $4.84\times$ at one enrichment
level, $1.17\times$ at the submitted pool, and $0.74$--$1.07\times$ once pools get rich. He
will write: *the order-of-magnitude claim is carried by a toy network and a synthetic family;
on the three road networks of the original submission the improvement is still a small factor,
and on most configurations there is none.*

*He will repeat the preprocessing objection verbatim, and it is not answered.* No reported
speedup charges the SVD or the nominal-flow computation. Our CSVs record SVD time (e.g. 4.6 s
at E2, 12.5 s at K15) but it is excluded from every ratio, and the nominal flow is still
obtained by solving the full problem. **This objection stands untouched.**

**On (2), validity of the accuracy evaluation.**

*Improved, but with a new exposure of our own making.* Negative gaps are gone, the reference is
a converged full-path gradient projection at certificate $9.1\times10^{-10}$, and Table~2
documents it. However, he asked specifically for *"the $\tau=0$ (uncompressed) problem solved
to sufficient convergence"* so that degradation at $\tau>0$ is purely compression. The draft
instead uses the best feasible point over operators. Worse, our own verification found the
C\texttt{++} uncompressed solver under-converging on grids by $4.6$--$4.9\%$ relative to
Python. A careful referee who reads the grid speedups will ask what the uncompressed baseline
converged to — and the honest answer is "not as far as Python's." **The grid headline is
exposed here.**

**On (3), accuracy and link-level error distribution.**

*Half-answered.* At the corrected rank the Chicago Sketch gaps are $0.02$--$0.03\%$ with
$R^2$ between $0.9997$ and $1.0000$ — a genuine, large improvement over the $7$--$8\%$ he
objected to. But he asked explicitly for *the distribution of link-level errors*, because
network design depends on per-link level of service. The draft still reports only aggregate
$R^2$ and objective gap. The machinery exists in `v3_metrics`. **Not done; he will notice.**

**On (4), dependence on the nominal flow.**

*This is now worse for us, not better.* We ran an exact ablation and proved the affine offset
is load-bearing: remove it and the mean grid speedup falls $2.60\times\to1.07\times$ and Sioux
falls $6.2\times\to1.34\times$. He objected that the method "depends entirely on the nominal
flow"; we have now *demonstrated* that dependence quantitatively, while still obtaining $w_0$
by solving the full problem. Unless paired with evidence that a cheap or stale $w_0$ suffices,
this hands him the argument. **Highest-severity open item.**

*Non-uniqueness of path flows* — that the major/minor split depends on which solution the
solver happens to return — is not addressed anywhere in the draft.

---

## Referee 2 — likely re-review

**On (1), motivation and application setting.**

*Better answered than before, though perhaps not where he will look.* He asked "in which
setting is path-space compression necessary?" The draft can now answer precisely: when
$2r$ is at most the average path length, when pools are rich, and when the anchor is reusable.
That is a real answer to his question — but it is delivered as a cost model in the numerical
section, not as a motivating argument in the introduction. He will likely still record the
motivation as unresolved unless the framing moves forward in the paper. His two other points —
that the convex Beckmann setting is restrictive relative to VI formulations, and that a full
path set is often unavailable — remain untouched.

**On (2), methodological contribution.**

*Rank: answered. $\tau$: not answered. Algorithm: contested.* He asked for analytical guidance
on $r$; the draft now gives a usable rule with a mechanism behind it, and shows the accuracy
is flat to four decimals across the useful range. That is exactly the kind of guidance he
requested. On $\tau$ he asked whether it should be uniform or vary by OD pair and how it
relates to convergence — the draft still uses a fixed quantile and says nothing analytic.
On his "the augmented Lagrangian is quite standard" complaint, the draft now shows the
pairing is *necessary* — compression accelerates ALM ($1.1$--$3.9\times$) and actively harms
projection and reduced-gradient methods ($0.00$--$0.34\times$), because compression destroys
the separable projection they rely on. That is a genuine contribution about *why* this
operator, but it is an explanation, not a new algorithm. He may accept it as a contribution
or may repeat that he wanted a tailored method; it is genuinely uncertain.

**On (3), benchmark comparisons.**

*Partly answered, and still short of what he asked.* The draft now compares ALM, Frank–Wolfe,
gradient projection and a reduced-gradient method under a matched certificate — far more than
the single OpenDTA reference he criticized. But he asked for *state-of-the-art path-based
approaches*, and the natural reading includes bush-based methods (Algorithm B, OBA, iTAPAS),
which remain the practical benchmark for static UE and appear nowhere. **He will repeat this.**

---

## Associate Editor — likely summary

The AE's rejection rested on three legs: no compelling computational advantage, unjustified
application setting, and invalid numerical validation. The draft moves the first substantially
(with a caveat about which instances carry it), moves the third substantially, and moves the
second only inside the numerical section. A plausible AE verdict on the current draft:
*major revision at a specialized journal; still short of OR* — unless the preprocessing
accounting and the nominal-flow circularity are answered, because those two go to whether the
speedup exists at all in the intended use case.

---

## Our position (author view)

We are not where the reviewers are, and it is worth being explicit about where we differ.

1. **We think the conditional result is the contribution.** The reviewers wanted an
   unconditional order-of-magnitude speedup. What we have is a characterized regime with a
   predictive rule, plus a demonstration that the benefit is operator-specific. That is a
   smaller claim but a more durable one, and it is honest.
2. **We agree with R1's mechanism and have made it ours.** His dense-block diagnosis is
   correct and now drives the design rule. This is common ground, not a dispute.
3. **We disagree that the evaluation is invalid**, but we accept that the *grid* baseline is
   not yet converged well enough to carry a headline number.
4. **We accept the preprocessing and nominal-flow objections as currently unanswered.** They
   are the reason the paper is not ready, not a matter of framing.

---

## What to do, in priority order

1. **The $w_0$ staleness experiment (highest value, not yet run).** Solve one instance,
   perturb the demand or capacities, and reuse the *old* $w_0$ on the perturbed instance.
   If the speedup survives a realistic perturbation, the repeated-solve motivation is proven
   and R1-4 collapses; if it does not, we have learned the method's true scope. This single
   experiment addresses R1-1 (preprocessing amortization), R1-4 (circularity), and R2-1
   (application setting) simultaneously. Nothing else has that leverage.
2. **Charge preprocessing explicitly.** Report every speedup twice: online-only and
   including SVD plus nominal flow, with an amortization count ("break-even after $k$
   related solves"). Cheap; removes a standing objection.
3. **Fix the uncompressed baseline on the grid**, then re-report the grid numbers against a
   properly converged reference. Until then the $6.8$--$8.6\times$ figures are vulnerable.
4. **Report the link-level error distribution** (percentiles of relative link-flow error).
   The machinery exists; R1 asked for it explicitly.
5. **Add a bush-based benchmark** (Algorithm B or iTAPAS) on one network, or state plainly
   why the comparison is not applicable.
6. **Give $\tau$ the same treatment $r$ received** — either an analytic rule or an honest
   statement that it is chosen by quantile and why that suffices.
7. **Move the "when does this pay" argument into the introduction.** It is currently the
   answer to R2's central question, buried in Section 6.
