# FW / Origin-Compatible Compression: Recorded Quick Results

This extension tests compression **inside feasible Frank--Wolfe and
origin-decomposed coordinate systems**, rather than replacing their linear
minimization oracles with a signed latent solve.

## Controlled route-rich Frank--Wolfe

Twenty OD blocks, exact BPR line search, common stopping rule, and exact demand
feasibility at every iterate.

| K | Full paths | Grouped atoms | Grouped objective gap | Grouped speedup | Storage reduction |
|---:|---:|---:|---:|---:|---:|
| 8 | 160 | 100 | 0.190% | 1.18x | 1.6x |
| 16 | 320 | 100 | 0.114% | 2.10x | 3.2x |
| 32 | 640 | 100 | 0.045% | 5.17x | 6.4x |

The route-rich gradient is direct: the number of grouped variables stays fixed
while the full pool grows with K.  The grouped model also requires fewer FW
iterations in the richer cases.

## Flow-weighted screening for standard FW

A signed SVD vector is not inserted into the FW simplex.  Instead, weighted
pivoted QR selects **actual paths**, after which ordinary restricted-pool FW is
used.

At K=16:

- unweighted screened objective gap: 0.308%;
- flow-weighted screened objective gap: 0.015%;
- weighted screened speedup vs full FW: 1.53x.

At K=32:

- unweighted screened objective gap: 0.245%;
- flow-weighted screened objective gap: 0.020%;
- weighted screened speedup vs full FW: 1.12x.

Thus flow weighting transfers to FW as a **feasible pool-screening mechanism**.
Periodic exact shortest-path pricing can restore global convergence in a
production column-generation implementation.

## Controlled origin-block conditional gradient

Four origins, five destinations per origin, and origin-specific link-flow
blocks.

| K | Full atoms | Grouped atoms | Grouped objective gap | Origin-block speedup | Storage reduction |
|---:|---:|---:|---:|---:|---:|
| 8 | 160 | 100 | 0.190% | 1.14x | 1.6x |
| 16 | 320 | 100 | 0.114% | 1.73x | 3.2x |
| 32 | 640 | 100 | 0.044% | 5.21x | 6.4x |

This demonstrates the compatibility interface for origin/bush methods:
compression remains OD-local within each origin, conservation is exact, and a
small number of path/branch families can replace a growing explicit route pool.

## Real-network checks

The same code runs on the packaged Sioux Falls and Chicago Sketch path sets.
The extracted sparse-support cases often converge in only a few FW iterations,
so their Python wall times are too small for strong solver claims; they are
primarily feasibility and transfer tests.  The C++ kernel gives the cleaner
implementation-level check.

## Standalone C++17 kernel

Recorded single-thread median-of-five forward/adjoint/LMO benchmark:

- Sioux Falls K=32: full 96 rows vs 15 grouped rows, **3.10x** kernel speedup.
- Chicago Sketch K=32: full 96 rows vs 15 grouped rows, **1.68x** kernel speedup.

Absolute times are hardware-dependent.  Row counts, feasibility, objective gaps,
and storage reductions are deterministic.

## Broader applicability

- Conjugate FW / bi-conjugate FW: retain history directions in atom coordinates.
- Origin/bush methods: use OD-local route/branch atoms and promote a branch when
  pricing or rollout identifies a missing direction.
- Multi-class assignment: separate class-OD atom blocks.
- ODME and repeated scenarios: reuse atom-link signatures and selected pools.
- Dynamic assignment: use feasible space-time path or schedule atoms inside a
  DNL/VI or column-generation framework; static FW alone is not sufficient for
  spillback and queue propagation.
