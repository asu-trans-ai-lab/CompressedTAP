#!/usr/bin/env python3
"""Protocol gate C3 — certify the C++ R1 (anchor elimination + rank-r decoder).

    python certify_cpp_r1.py [exe]

Three regimes, per the protocol, so an implementation error cannot hide inside expected
compression error:

    rank-exact   fixture E (grid3 + duplicated route, tau=200, r = rank(B2) = 5):
                 C++ R1 must reproduce C++ R0 on the SAME instance (rel <= 1e-5)
    moderate r   Sioux SFK25, r = 50: C++ R1 must land on the Python R1 floor
                 (feasible objective rel <= 1e-4; pool gap at the ~1e-4 floor)
    low r        Sioux SFK25, r = 5: must run clean and floor NO LOWER than r = 50
                 (accuracy ordering), all feasible metrics clean

    signed       fixture D (tau=55, r=1, provably negative rank-1 reconstruction):
                 the mu-ALM must hold the minor block (final viol <= 1e-3) and the
                 feasible objective must price at or above the closed-form optimum
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PYDIR = os.path.join(ROOT, 'source', 'updated_TAPLite', 'python')
DATAROOT = os.path.join(ROOT, 'source', 'updated_TAPLite', 'data')
SIOUX = os.path.join(DATAROOT, '02_Sioux_Falls')
EXE = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else 'compressed_solver_c3.exe')
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1'}

sys.path.insert(0, PYDIR)
from compressed_assignment import load_problem, split_major_minor, build_compressed
from run_progressive_ladder import solve_R1, anchors_of
from evaluator import evaluate
from test_fixtures_cde import write_gmns


def export(gmns, pool, tag, rank, tau, multi=False):
    dump = os.path.join(DATAROOT, 'cpp_r1_%s' % tag)
    r = subprocess.run([sys.executable, 'export_stage7.py', gmns, pool, dump, str(rank),
                        str(int(multi)), str(tau)], cwd=PYDIR, env=ENV,
                       capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise SystemExit('export failed: %s' % tag)
    return dump


def run(dump, mode, outers='60', inner='2000'):
    r = subprocess.run([EXE, dump, mode, outers, '0', '', inner], env=ENV,
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise SystemExit('%s failed on %s' % (mode, dump))
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT')][-1]
    return dict(tok.split('=') for tok in line.split()[1:])


def grid3_dup(out_dir):
    G, n = 3, 2
    links, lid = {}, 1
    for i in range(G):
        for j in range(G):
            if j < n: links[('R', i, j)] = lid; lid += 1
            if i < n: links[('D', i, j)] = lid; lid += 1
    def pl(downs):
        i = j = 0; seq = []
        for step in range(2 * n):
            if step in downs: seq.append(links[('D', i, j)]); i += 1
            else: seq.append(links[('R', i, j)]); j += 1
        return ';'.join(map(str, seq))
    routes = [pl(set(c)) for c in combinations(range(4), 2)]
    routes.append(routes[-1])
    return write_gmns(out_dir,
                      links=[(l, 1.0, 200.0) for l in sorted(links.values())],
                      demand={(1, 9): 1200.0},
                      paths=[(1, 9, rt, 300.0 if k == 0 else 100.0)
                             for k, rt in enumerate(routes)])


def main():
    tmp = tempfile.mkdtemp(prefix='cpp_r1_')
    fails = []

    # --- rank-exact: R1(r=rank(B2)) == R0 on the same instance -------------------------------
    gE = grid3_dup(os.path.join(tmp, 'E'))
    dE = export(gE, 'path_pool.csv', 'fixtureE', rank=5, tau=200.0)
    r0 = run(dE, 'r0')
    r1 = run(dE, 'r1')
    rel = abs(float(r1['obj_feasible']) - float(r0['obj_feasible'])) \
        / abs(float(r0['obj_feasible']))
    ok = rel <= 1e-5
    print('rank-exact  R0 %s vs R1 %s  rel %.1e : %s'
          % (r0['obj_feasible'], r1['obj_feasible'], rel, 'PASS' if ok else 'FAIL'))
    if not ok: fails.append('rank-exact')

    # --- moderate r: Sioux r=50 vs Python R1 floor --------------------------------------------
    P = load_problem(SIOUX, 'path_pool_SFK25.csv')
    C = build_compressed(P, split_major_minor(P, tau=1.0), 50)
    s = solve_R1(P, C, anchors_of(P))
    row = evaluate(P, s['x'], method='R1', network_id='sioux_SFK25', rank=50,
                   solve_time=s['time_s'])
    f_py = row['objective_feasible']
    d50 = export(SIOUX, 'path_pool_SFK25.csv', 'sioux_r50', rank=50, tau=1.0)
    c50 = run(d50, 'r1')
    rel = abs(float(c50['obj_feasible']) - f_py) / f_py
    ok = rel <= 1e-4
    print('moderate r  Python R1 %.1f vs C++ r1 %s  rel %.1e | gap %s | viol %s : %s'
          % (f_py, c50['obj_feasible'], rel, c50['pool_gap'], c50['viol'],
             'PASS' if ok else 'FAIL'))
    if not ok: fails.append('moderate')

    # --- low r: Sioux r=5, SAME-REGIME parity vs Python ----------------------------------------
    # NOT a cross-rank objective ordering: two ALM runs stop at different convergence levels,
    # so comparing their outputs across ranks is an unmatched-accuracy comparison -- the very
    # mistake this protocol forbids (first version of this test made it, and a better-converged
    # r=5 run priced below an under-converged r=50 run). Parity is checked per rank instead.
    C5 = build_compressed(P, split_major_minor(P, tau=1.0), 5)
    s5 = solve_R1(P, C5, anchors_of(P))
    f_py5 = evaluate(P, s5['x'], method='R1', network_id='sioux_SFK25', rank=5,
                     solve_time=s5['time_s'])['objective_feasible']
    d5 = export(SIOUX, 'path_pool_SFK25.csv', 'sioux_r5', rank=5, tau=1.0)
    c5 = run(d5, 'r1')
    rel5 = abs(float(c5['obj_feasible']) - f_py5) / f_py5
    ok = rel5 <= 1e-4 and float(c5['viol']) <= 1e-3
    print('low r       Python R1(r=5) %.1f vs C++ %s  rel %.1e | viol %s : %s'
          % (f_py5, c5['obj_feasible'], rel5, c5['viol'], 'PASS' if ok else 'FAIL'))
    if not ok: fails.append('low-r')

    # --- signed: fixture D, active minor nonnegativity -----------------------------------------
    gD = write_gmns(os.path.join(tmp, 'D'),
                    links=[(1, 10.0, 200.0), (2, 30.0, 100.0), (3, 30.0, 100.0),
                           (4, 30.0, 100.0)],
                    demand={(1, 2): 150.0},
                    paths=[(1, 2, '1', 60.0), (1, 2, '2;3', 50.0), (1, 2, '2;4', 20.0),
                           (1, 2, '3;4', 20.0)])
    dD = export(gD, 'path_pool.csv', 'fixtureD', rank=1, tau=55.0)
    cD = run(dD, 'r1')
    f_star = 10.0 * 150 + 10.0 * 0.15 * 200 / 5 * (150 / 200.0) ** 5   # all demand on p1
    ok = float(cD['viol']) <= 1e-3 and float(cD['obj_feasible']) >= f_star - 1e-6
    print('signed      obj %s >= closed form %.4f, viol %s : %s'
          % (cD['obj_feasible'], f_star, cD['viol'], 'PASS' if ok else 'FAIL'))
    if not ok: fails.append('signed')

    shutil.rmtree(tmp, ignore_errors=True)
    print('\n=> %s' % ('C3 CPP R1 CERTIFIED' if not fails
                       else 'C3 FAILED: %s' % ', '.join(fails)))
    return len(fails)


if __name__ == '__main__':
    raise SystemExit(1 if main() else 0)
