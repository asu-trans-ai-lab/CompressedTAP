#!/usr/bin/env python3
"""Protocol gate C2 — certify the C++ R0 (exact capped-simplex SPG) against Python R0.

    python certify_cpp_r0.py

Networks, per the protocol's C2 list:
    two_corridor   fixture A: closed-form anchor (also certified against the analytic UE)
    fixtureC       ACTIVE anchor cap: the anchor carries zero equilibrium flow, so the cap
                   binds at the solution — the projection must actually work, not idle
    grid3          complete 6-route set
    sioux_SFK25    the manuscript's formulation-comparison network

For each: export the frozen instance, run `compressed_solver.exe r0`, run Python solve_R0,
and compare f(v), v, the path-pool relative gap, and ||Ax - d||_inf. Python R0 enforces the
anchor bound by ALM penalty (approximate, <=1e-3); the C++ R0 enforces it EXACTLY by
projection — so the C++ side must be at least as feasible, and objectives must agree to the
convergence tolerance.
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PYDIR = os.path.join(ROOT, 'source', 'updated_TAPLite', 'python')
DATAROOT = os.path.join(ROOT, 'source', 'updated_TAPLite', 'data')
TWO_CORRIDOR = os.path.join(ROOT, 'stable_release', '00_two_corridor', 'data')
SIOUX = os.path.join(DATAROOT, '02_Sioux_Falls')
EXE = os.path.join(HERE, 'compressed_solver.exe')
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1'}

sys.path.insert(0, PYDIR)
from compressed_assignment import load_problem
from run_progressive_ladder import solve_R0, anchors_of
from evaluator import project_simplex_od, total_link_flow, pool_relgap_feasible
from test_fixtures_cde import write_gmns
from kernel_parity import make_grid3


def run_cpp_r0(gmns_dir, pool, tag, multi=False, iters=60000):
    dump = os.path.join(DATAROOT, 'cpp_r0_%s' % tag)
    r = subprocess.run([sys.executable, 'export_stage7.py', gmns_dir, pool, dump, '0',
                        str(int(multi))], cwd=PYDIR, env=ENV, capture_output=True,
                       text=True, timeout=1200)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise SystemExit('export failed: %s' % tag)
    vfile = os.path.join(dump, 'v_r0.f64')
    r = subprocess.run([EXE, dump, 'r0', str(iters), '0', vfile], env=ENV,
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise SystemExit('r0 failed: %s' % tag)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT')][-1]
    kv = dict(tok.split('=') for tok in line.split()[1:])
    return np.fromfile(vfile, dtype=np.float64), kv


def main():
    tmp = tempfile.mkdtemp(prefix='cpp_r0_')
    nets = []
    nets.append(('two_corridor', TWO_CORRIDOR, 'path_pool.csv'))
    fixC = write_gmns(os.path.join(tmp, 'C'),
                      links=[(1, 10.0, 100.0), (2, 10.0, 100.0), (3, 50.0, 100.0)],
                      demand={(1, 2): 150.0},
                      paths=[(1, 2, '1', 25.0), (1, 2, '2', 25.0), (1, 2, '3', 100.0)])
    nets.append(('fixtureC_active_cap', fixC, 'path_pool.csv'))
    nets.append(('grid3', make_grid3(os.path.join(tmp, 'g3')), 'path_pool.csv'))
    nets.append(('sioux_SFK25', SIOUX, 'path_pool_SFK25.csv'))

    n_fail = 0
    for tag, gmns, pool in nets:
        P = load_problem(gmns, pool)
        s = solve_R0(P, anchors_of(P))
        x_py = project_simplex_od(s['x'], P['d'], P['p2od'], P['n_od'])
        v_py = total_link_flow(P, x_py)
        f_py = float(P['bpr'].beckmann(v_py))
        g_py = pool_relgap_feasible(P, x_py)

        v_c, kv = run_cpp_r0(gmns, pool, tag)
        f_c, g_c = float(kv['obj_feasible']), float(kv['pool_gap'])
        cons_c, minanc = float(kv['cons']), float(kv['min_anchor'])

        rel_f = abs(f_c - f_py) / max(1.0, abs(f_py))
        rel_v = float(np.max(np.abs(v_c - v_py)) / max(1.0, float(np.max(np.abs(v_py)))))
        ok = (rel_f <= 1e-6 and rel_v <= 1e-3 and cons_c <= 1e-9 * max(P['d'].max(), 1.0)
              and minanc >= -1e-12 and g_c <= max(2.0 * max(g_py, 0.0), 1e-5))
        print('%-22s f_py %14.4f  f_cpp %14.4f (rel %.1e) | v rel %.1e | '
              'gap py %.1e cpp %.1e | cons %.1e | min_anchor %+.1e | anchors@0 %s : %s'
              % (tag, f_py, f_c, rel_f, rel_v, g_py, g_c, cons_c, minanc,
                 kv['anchors_at_bound'], 'PASS' if ok else 'FAIL'), flush=True)
        if not ok:
            n_fail += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print('\n=> %s' % ('C2 CPP R0 CERTIFIED' if n_fail == 0
                       else 'C2 FAILED on %d network(s)' % n_fail))
    return n_fail


if __name__ == '__main__':
    raise SystemExit(1 if main() else 0)
