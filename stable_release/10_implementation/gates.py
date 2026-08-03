"""Mandatory consistency gates for a completed (network, pool) scenario.
Every solver that enters a comparison must pass, pairwise against the
best-objective solution of the set:

  G1 link-volume consistency: E_v <= 1e-3 and R_v^2 >= 0.9999
  G2 link-time (BPR) consistency: E_t <= 1e-3
  G3 Beckmann objective reference consistency: relative spread <= 1e-5

usage: python gates.py <bpr_dump_dir> <name:v_file.f64> [<name:v_file> ...]
"""
import os
import sys

import numpy as np

dd = sys.argv[1]
t0 = np.fromfile(os.path.join(dd, 'bpr_t0.f64'))
al = np.fromfile(os.path.join(dd, 'bpr_alpha.f64'))
be = np.fromfile(os.path.join(dd, 'bpr_beta.f64'))
cap = np.fromfile(os.path.join(dd, 'bpr_cap.f64'))

def times(v):
    return t0 * (1.0 + al * np.power(np.maximum(v, 0.0) / cap, be))

def beck(v):
    vp = np.maximum(v, 0.0)
    return float(np.sum(t0 * vp + t0 * al * cap / (be + 1.0)
                        * np.power(vp / cap, be + 1.0)))

sols = []
for spec in sys.argv[2:]:
    name, path = spec.split(':', 1)
    v = np.fromfile(path)
    sols.append((name, v, beck(v)))
ref_name, ref_v, ref_F = min(sols, key=lambda s: s[2])
ref_t = times(ref_v)
print(f'reference = {ref_name} (lowest Beckmann {ref_F:,.2f})')
print(f"{'solver':10s} {'E_v':>10s} {'R_v^2':>10s} {'E_t':>10s} "
      f"{'dF/F':>11s}  gates")
ok_all = True
for name, v, F in sols:
    ev = float(np.linalg.norm(v - ref_v) / np.linalg.norm(ref_v))
    r2 = float(1 - np.sum((v - ref_v) ** 2)
               / max(np.sum((ref_v - ref_v.mean()) ** 2), 1e-30))
    tv = times(v)
    et = float(np.linalg.norm(tv - ref_t) / np.linalg.norm(ref_t))
    df = (F - ref_F) / ref_F
    g1 = ev <= 1e-3 and r2 >= 0.9999
    g2 = et <= 1e-3
    g3 = df <= 1e-5
    ok = g1 and g2 and g3
    ok_all &= ok
    print(f'{name:10s} {ev:10.2e} {r2:10.6f} {et:10.2e} {df:11.2e}  '
          f'{"PASS" if ok else "FAIL"}'
          f'{"" if ok else "  (G1 " + str(g1) + " G2 " + str(g2) + " G3 " + str(g3) + ")"}')
print('GATES:', 'ALL-PASS' if ok_all else 'FAILURES-PRESENT')
