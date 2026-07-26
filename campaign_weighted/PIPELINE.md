# The numerical pipeline, end to end — what each stage does, costs, and what we learned

One pass through the method, in the order the code executes it, with the measured cost of each
stage and the lesson that stage taught during this campaign. Numbers are from the grid
$N{=}32,K{=}64$ instance (66,161 paths) unless stated otherwise.

---

## Stage 0 — candidate path pool

**What:** a pool of candidate paths per OD pair, from penalty $K$-shortest-path
(`ksp_gen.exe`, deterministic: shortest path, then $K$ rounds of multiplying the last path's
link costs by 1.4 and re-solving).

**Cost:** seconds to a few minutes; not on the online path.

**Lessons.**
- *Never cache a pool.* A stale pool silently corrupts the threshold and the nominal flow, and
  produces a table that looks plausible. This happened once and the run had to be discarded.
- *Pool richness is the primary axis*, not network size. $\bar K = n/\ell$ determines whether
  compression helps at all.
- *KSP extras carry no flow.* Without a nominal flow assigned to them, the affine box
  $w_0 + U_r z \ge 0$ collapses to $\{0\}$ and the compressed master degenerates to the majors.
  A logit split of demand over the pool by base cost fixes this and is required, not optional.

## Stage 1 — nominal flow $w_0$

**What:** solve the uncompressed problem once; its path flows are the nominal flow.

**Cost: 65.22 s — 99% of all preprocessing.**

**Lessons.**
- *This is the whole preprocessing cost.* The SVD is 1%. Any discussion of "is preprocessing
  worth it" is a discussion about this solve, not about the factorisation.
- *It is also the circularity.* Compressing a single instance requires solving it first, so the
  method only makes sense across related solves — which is why the break-even count exists.
- *The offset built from it is load-bearing.* Removing $w_0$ raises the mean grid gap from
  1.112% to 1.394% and collapses the best cells; it improves solve time beyond the noise floor
  in 10 of 20 cells. It is a speed mechanism, not only an accuracy one.

## Stage 2 — major/minor split

**What:** paths with nominal flow above a threshold $\tau$ are kept explicit (majors, $s$ of
them); the rest are compressed. $\tau$ is set by a quantile of $w_0$ (75th on the grid).

**Cost:** negligible.

**Lessons.**
- *This sets the reduction*, and reduction is what the speedup tracks — 86.4% here.
- *A fixed quantile misfires on thin pools.* At $K=4$ it classifies genuinely used paths as
  minor and costs 1.4–3.4% objective gap. Rich pools are forgiving; thin ones are not.
- *No analytical rule for $\tau$ exists yet.* This is the one parameter the study still chooses
  empirically, and a reviewer asked for exactly that.

## Stage 3 — the compressed basis $\Phi_r$

**What:** factorise $S B_2$ with $S=\operatorname{diag}(\sqrt{w_0}+\varepsilon)$ and take the
$r$ leading left singular vectors as $\Phi_r$. Minor flows become $x_{\mathcal M}=w_0+\Phi_r z$.
Then form the dense blocks $D = B_2^\top\Phi_r$ and $M = A_2\Phi_r$.

**Cost: 0.76 s.**

**Lessons.**
- *Rank is a speed parameter, not an accuracy parameter.* Over $r\in\{10,20,50\}$ the speedup
  moves by nearly an order of magnitude while the feasible gap is unchanged to four decimal
  places. Choose $2r \lesssim$ links-per-path; the optimum is interior.
- *The row scaling is a heuristic, and we say so.* It is not the minimiser of the flow-weighted
  reconstruction loss — that would be $\operatorname{span}(S^{-1}\widetilde U_r)$. Measured at
  $r=20$: 249.1 as implemented, 209.1 for the true minimiser, 227.9 unweighted. It is kept
  because it improves the *solved* objective gap in 27 of 34 paired comparisons.
- *Zero nominal flows are a regime, not an edge case.* 80.6% of Sioux minor paths have
  $w_0=0$; Chicago Sketch has none. The two behave completely differently.

## Stage 4 — the compressed solve

**What:** SPG-ALM on $(y,z)$: OD conservation by multipliers, minor nonnegativity
$w_0+\Phi_r z\ge0$ by an augmented-Lagrangian hinge.

**Cost: 14.1 s** against 50.1 s warm-started uncompressed and 64.3 s cold.

**Lessons.**
- *The dense reconstruction is the binding cost.* Each gradient pays about $2r$ dense
  operations per minor path, against the sparsity of a path in the full problem. This single
  fact predicts every speedup result in the study.
- *The benefit is ALM-specific.* Compression accelerates ALM (1.1–3.9×) and actively harms
  gradient projection (0.00–0.10×) and reduced gradient (0.07–0.34×), because it destroys the
  separable per-OD projection those methods depend on. Variable reduction helps a method whose
  bottleneck is in the variables, not in the constraints.
- *The engine changes the answer.* Python (BLAS-fast dense) and C++ (fast sparse loops) give
  different — sometimes opposite — speedup curves on the same instances. Ratios are only
  reported within an implementation.

## Stage 5 — recovery and evaluation

**What:** reconstruct $x$, project per OD onto $\{x\ge0,\ Ax=d\}$, evaluate the Beckmann
objective and link flows.

**Lessons.**
- *Always compare after the same conversion.* $\delta_F$ (the $\ell_1$ mass the projection
  moves) is a property of the solver's terminal iterate, not of the compression, and differs
  between engines on the same instance.
- *Objectives are exact; times are not.* Objectives reproduce bit for bit; identical solves
  vary 1.1–2.8× in wall time. Hence the 1.45× noise floor and the rule that no smaller speed
  difference is claimed.
- *A good objective can hide a worse link solution.* In the repeated-scenario experiment the
  compressed solve attains 0.0000% objective gap but 5.45% link-flow error against ~2% for the
  warm-started full solve. Report both.

---

## Reuse: the setting the pipeline is for

Stages 0–3 are paid once. Stages 4–5 are paid per scenario. With
$T_{\text{prep}} = 65.98$ s and a saving of $35.94$ s per follow-up over a warm-started full
solve,

    H* = ceil( 65.98 / 35.94 ) = 2 related solves to repay preprocessing,

and $H^\ast=1$ when the baseline solve already exists. Below that count, a warm-started
uncompressed solve is the better choice, and the method does not claim otherwise.

## The five transferable lessons

1. **Variable reduction only helps if the bottleneck is in the variables.** Compression
   densifies constraints, so it accelerates ALM and harms projection-type methods.
2. **A signed subspace of nonnegative flows needs an interior anchor.** $w_0$ supplies it, and
   removing it costs both accuracy and speed.
3. **Rank is a cost dial with an interior optimum**, at roughly $2r \lesssim$ links-per-path.
4. **Speedup is a property of (representation × operator × implementation × instance),** never
   of "compression" alone.
5. **Separate the exact from the noisy.** Objectives are deterministic; timings are not. State
   the noise floor and refuse claims below it — including your own earlier ones.
