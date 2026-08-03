"""The honest gap (Xuesong's discipline, 2026-07-12): recover the explicit
path-flow solution from the latent state, form the recovered link flows,
and measure the TRUE user-equilibrium relative gap by INDEPENDENT full
pricing --- never the solver's internal reduced-space accounting.

For the feasible-column (nonnegative atom) method recovery is exact:
x = mu*pi >= 0 and v = B'x exactly, so v is the solver's link-flow dump.
This evaluator touches only the pool (B, p2od, demand), the BPR params,
and v; it shares no code with the solver, so a match certifies that the
reported number is a genuine recovered-flow UE gap, not a self-graded one.

gap = TSTT/SPTT - 1,  TSTT = sum_a v_a t_a(v),
      SPTT = sum_w q_w * min_{k in pool(w)} d_k,  d_k = sum_a delta_ak t_a(v).

usage: python true_recovered_gap.py <cpp_problem_dir> <v_dump.f64>
"""
import os
import sys

import numpy as np

dd, vf = sys.argv[1], sys.argv[2]
t0 = np.fromfile(dd + '/bpr_t0.f64'); al = np.fromfile(dd + '/bpr_alpha.f64')
be = np.fromfile(dd + '/bpr_beta.f64'); cap = np.fromfile(dd + '/bpr_cap.f64')
Bp = np.fromfile(dd + '/B_indptr.i64', dtype=np.int64)
Bi = np.fromfile(dd + '/B_indices.i32', dtype=np.int32)
p2od = np.fromfile(dd + '/p2od.i32', dtype=np.int32)
d = np.fromfile(dd + '/dvec.f64')
v = np.fromfile(vf)

t = t0 * (1.0 + al * np.power(np.maximum(v, 0.0) / cap, be))
npath = len(Bp) - 1
tcum = np.concatenate([[0.0], np.cumsum(t[Bi])])
dk = tcum[Bp[1:]] - tcum[Bp[:-1]]            # path costs at recovered costs
best = np.full(d.size, np.inf)
np.minimum.at(best, p2od, dk)
SPTT = float((d * best).sum())
TSTT = float((v * t).sum())
gap = TSTT / SPTT - 1.0
print(f'{os.path.basename(vf)}: TRUE recovered-flow UE gap = {gap:.4e}  '
      f'(TSTT {TSTT:,.1f}  SPTT {SPTT:,.1f})')
