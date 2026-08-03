#!/usr/bin/env python3
"""Protocol gate C1 — C++ kernel parity against the Python reference implementation.

    python kernel_parity.py

For each basic network, freezes deterministic test vectors x (including one with NEGATIVE
entries, so the BPR objective/gradient consistency below v = 0 — the fixed defect class — is
actually exercised, not just the feasible region), computes every primitive in the Python
reference (compressed_assignment), evaluates the same primitives in C++
(compressed_solver.exe kernels mode) from the same frozen instance blobs and vectors, and
compares component-wise to the protocol tolerances:

    quantity        tolerance (relative, worst component)
    link flows  v   1e-12
    BPR costs   t   1e-12
    objective       1e-11
    gradient  B t   1e-10
    pool relgap     1e-10

Also runs the P1 finite-difference gradient check in Python (central differences at a point
where some link flows are NEGATIVE) — the regression that catches M4-class objective defects.

Networks: two_corridor (closed-form anchor), two_corridor_singleton (v0 background), grid3
(complete 6-route set), sioux_SFK25. Writes kernel_parity.csv; gate line 'C1 KERNEL PARITY
PASSED' only if every row passes.
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
from compressed_assignment import load_problem, project_feasible, pool_relgap

TOLS = dict(v=1e-12, t=1e-12, obj=1e-11, g=1e-10, gap=1e-10)


def make_grid3(out_dir):
    """3x3 Manhattan grid, one OD, the complete 6 monotone routes (fixture from the
    multi-network suite, duplicated here so this gate is self-contained)."""
    from itertools import combinations
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
        return seq
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'link.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['link_id', 'from_node_id', 'to_node_id', 'length', 'vdf_length_mi',
                    'capacity', 'free_speed', 'vdf_free_speed_mph', 'vdf_alpha', 'vdf_beta'])
        for (kind, i, j), l in sorted(links.items(), key=lambda kv: kv[1]):
            w.writerow([l, i * G + j + 1, (i + (kind == 'D')) * G + j + (kind == 'R') + 1,
                        1.0, 1.0, 200.0, 60.0, 60.0, 0.15, 4.0])
    with open(os.path.join(out_dir, 'demand.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['o_zone_id', 'd_zone_id', 'volume'])
        w.writerow([1, 9, 1200.0])
    with open(os.path.join(out_dir, 'path_pool.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['o_zone_id', 'd_zone_id', 'link_ids', 'prob_ref', 'volume_ref',
                    'cost_base', 'source', 'path_id'])
        for p, downs in enumerate(combinations(range(4), 2)):
            w.writerow([1, 9, ';'.join(map(str, pl(set(downs)))), 1 / 6, 200.0, 4.0,
                        'gen', float(p)])
    return out_dir


def make_singleton_variant(tmp_root):
    tmp = os.path.join(tmp_root, 'two_corridor_singleton')
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(os.path.join(TWO_CORRIDOR, 'link.csv'), tmp)
    with open(os.path.join(TWO_CORRIDOR, 'demand.csv')) as f, \
         open(os.path.join(tmp, 'demand.csv'), 'w', newline='') as g:
        g.write(f.read().rstrip('\n') + '\n9,99,500.0\n')
    with open(os.path.join(TWO_CORRIDOR, 'path_pool.csv')) as f, \
         open(os.path.join(tmp, 'path_pool.csv'), 'w', newline='') as g:
        g.write(f.read().rstrip('\n') + '\n9,99,1,1.0,500.0,0.0,singleton_test,900.0\n')
    return tmp


def test_vectors(P, seed=7):
    rng = np.random.default_rng(seed)
    x0 = np.maximum(P['x0'], 1e-3).astype(float)
    V = {'typical': x0,
         'under': 0.5 * x0,
         'perturbed': x0 * (1.0 + 0.3 * rng.standard_normal(P['n'])),
         # negative entries -> some link flows go NEGATIVE: exercises the objective/gradient
         # extension below zero, where the fixed BPR defect class lives
         'signed': x0 - 0.8 * float(np.median(x0[x0 > 0]) + x0.max() * 0.2)}
    V['projected'] = project_feasible(V['perturbed'], P['A'], P['d'], P['p2od'])
    return V


def python_kernels(P, x):
    v = np.asarray(P.get('v0', 0.0) + P['B'].T @ x).flatten()
    t = P['bpr'].t(v)
    g = np.asarray(P['B'] @ t).flatten()
    return v, t, float(P['bpr'].beckmann(v)), g


def fd_gradient_check(P, x, k=12):
    """Central-difference check of dF/dx = B t(v) at x (which yields some v < 0).

    The step is scaled per coordinate: with f ~ 1e7 (Sioux) an absolute h = 1e-4 puts the
    two objective values within double-precision cancellation of each other, and the check
    measures rounding, not the gradient."""
    _, _, _, g = python_kernels(P, x)
    idx = np.linspace(0, P['n'] - 1, min(k, P['n'])).astype(int)
    errs = []
    for p in idx:
        h = 1e-4 * (1.0 + abs(float(x[p])))
        e = np.zeros(P['n']); e[p] = h
        fp = python_kernels(P, x + e)[2]
        fm = python_kernels(P, x - e)[2]
        errs.append(abs((fp - fm) / (2 * h) - g[p]) / max(1.0, abs(g[p])))
    return float(max(errs))


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(a)))))


def run_network(name, gmns_dir, pool, rank, multi, rows, tmp_root):
    dump = os.path.join(DATAROOT, 'cpp_parity_%s' % name)
    r = subprocess.run([sys.executable, 'export_stage7.py', gmns_dir, pool, dump,
                        str(rank), str(int(multi))], cwd=PYDIR, env=ENV,
                       capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise SystemExit('export failed: %s' % name)
    P = load_problem(gmns_dir, pool, multi_path_only=multi)

    # P1: FD gradient at the signed vector (some v < 0)
    V = test_vectors(P)
    fd = fd_gradient_check(P, V['signed'])
    rows.append(dict(network=name, vector='signed', quantity='fd_gradient',
                     max_rel_err='%.3e' % fd, tol='1e-06',
                     status='pass' if fd <= 1e-6 else 'FAIL'))

    for vname, x in V.items():
        xf = os.path.join(tmp_root, '%s_%s.f64' % (name, vname))
        np.ascontiguousarray(x, dtype=np.float64).tofile(xf)
        r = subprocess.run([EXE, dump, 'kernels', '0', '0', xf], env=ENV,
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(r.stdout, r.stderr); raise SystemExit('kernels failed: %s' % name)
        kline = [ln for ln in r.stdout.splitlines() if ln.startswith('KERNELS')][-1]
        kv = dict(tok.split('=') for tok in kline.split()[1:])
        v_c = np.fromfile(xf + '.v.f64', dtype=np.float64)
        t_c = np.fromfile(xf + '.t.f64', dtype=np.float64)
        g_c = np.fromfile(xf + '.g.f64', dtype=np.float64)
        v_p, t_p, obj_p, g_p = python_kernels(P, x)
        checks = [('v', rel_err(v_p, v_c)), ('t', rel_err(t_p, t_c)),
                  ('obj', abs(obj_p - float(kv['obj'])) / max(1.0, abs(obj_p))),
                  ('g', rel_err(g_p, g_c))]
        if vname == 'projected':   # gap formula parity, at a point where projection = identity
            checks.append(('gap', abs(pool_relgap(P, x) - float(kv['gap']))
                           / max(1e-12, abs(pool_relgap(P, x)))))
        for q, e in checks:
            rows.append(dict(network=name, vector=vname, quantity=q,
                             max_rel_err='%.3e' % e, tol='%.0e' % TOLS[q],
                             status='pass' if e <= TOLS[q] else 'FAIL'))


def main():
    tmp_root = tempfile.mkdtemp(prefix='kernel_parity_')
    rows = []
    run_network('two_corridor', TWO_CORRIDOR, 'path_pool.csv', 0, False, rows, tmp_root)
    run_network('two_corridor_singleton', make_singleton_variant(tmp_root),
                'path_pool.csv', 0, True, rows, tmp_root)
    run_network('grid3', make_grid3(os.path.join(tmp_root, 'grid3')),
                'path_pool.csv', 0, False, rows, tmp_root)
    run_network('sioux_SFK25', SIOUX, 'path_pool_SFK25.csv', 50, False, rows, tmp_root)

    out = os.path.join(HERE, 'kernel_parity.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    nf = sum(1 for r_ in rows if r_['status'] == 'FAIL')
    wide = {}
    for r_ in rows:
        wide.setdefault((r_['network'], r_['vector']), {})[r_['quantity']] = r_['max_rel_err']
    for (net, vec), q in wide.items():
        print('%-22s %-9s  ' % (net, vec)
              + '  '.join('%s %s' % (k, q[k]) for k in sorted(q)))
    print('\nsaved %s' % out)
    print('=> %s' % ('C1 KERNEL PARITY PASSED (%d checks)' % len(rows) if nf == 0
                     else 'C1 FAILED: %d of %d checks' % (nf, len(rows))))
    shutil.rmtree(tmp_root, ignore_errors=True)
    return nf


if __name__ == '__main__':
    raise SystemExit(1 if main() else 0)
