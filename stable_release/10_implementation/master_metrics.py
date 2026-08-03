"""Accuracy panel for the pool masters: link-volume R^2 / E_v and
link-time (BPR) R^2 / E_t of each master's terminal solution against the
CLASSICAL reference: GP run to a deeper tolerance than any compared run
(v_<tag>_gpref.f64; falls back to the ordinary gp dump if absent). The
new methods are never their own yardstick — consistent with the one-OD
protocol (deeply converged full-GP reference) and the paper's deep
tau=0 reference convention.

usage: python master_metrics.py sk|e0|rich
"""
import json
import os
import sys

import numpy as np

TAG = sys.argv[1]
DIRS = {'sk': 'cpp_problem_E2_full_rich',
        'e0': 'cpp_problem_E0_baseline_regional',
        'rich': 'cpp_problem_E2_rich_regional'}
dd = os.path.join('..', 'data', DIRS[TAG])
t0 = np.fromfile(os.path.join(dd, 'bpr_t0.f64'))
al = np.fromfile(os.path.join(dd, 'bpr_alpha.f64'))
be = np.fromfile(os.path.join(dd, 'bpr_beta.f64'))
cap = np.fromfile(os.path.join(dd, 'bpr_cap.f64'))

def times(v):
    return t0 * (1.0 + al * np.power(np.maximum(v, 0.0) / cap, be))

if os.path.exists(f'v_{TAG}_gpref.f64'):
    ref = np.fromfile(f'v_{TAG}_gpref.f64')
    rname = 'deep classical GP (gpref)'
else:
    ref = np.fromfile(f'v_{TAG}_gp.f64')
    rname = 'gp terminal (gpref pending)'
t_ref = times(ref)
print(f'[{TAG}] reference = {rname} ({ref.size:,} links)')
hdr = f"{'master':8s} {'R_v^2':>10s} {'E_v':>10s} {'R_t^2':>10s} {'E_t':>10s}"
print(hdr)
for m in ('latgp', 'gp', 'mlgp'):
    f = f'v_{TAG}_{m}.f64'
    if not os.path.exists(f):
        print(f'{m:8s}   (missing)')
        continue
    v = np.fromfile(f)
    tv = times(v)
    r2v = 1.0 - np.sum((v - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)
    ev = np.linalg.norm(v - ref) / np.linalg.norm(ref)
    r2t = 1.0 - np.sum((tv - t_ref) ** 2) / np.sum((t_ref - t_ref.mean()) ** 2)
    et = np.linalg.norm(tv - t_ref) / np.linalg.norm(t_ref)
    print(f'{m:8s} {r2v:10.6f} {ev:10.2e} {r2t:10.6f} {et:10.2e}')
print('note: link-time R^2/E_t are computed at BPR times of the terminal '
      'volumes; the metric formerly called "BPR-fit R^2" is R_t^2 here.')
