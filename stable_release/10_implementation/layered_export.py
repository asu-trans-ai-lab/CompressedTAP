"""Layered binary-choice network generator (tier-0 grid family): H stages
of 2 parallel links -> 2H links, 2^H paths, ONE OD pair. Because the pool
enumerates ALL paths, pool solvers and full-network solvers (Algorithm B)
face the identical problem. Emits cpp_problem_<tag>/ for pool_masters and
TNTP net/trips for tap-b.

usage: python layered_export.py <H> <q> <out_root>
"""
import json
import os
import sys

import numpy as np

H, Q, ROOT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
CAPQ = float(sys.argv[4]) if len(sys.argv) > 4 else Q
tag = f'layered{H}'
out = os.path.join(ROOT, f'cpp_problem_{tag}')
os.makedirs(out, exist_ok=True)

m = 2 * H
# per stage: link0 (fast) t0=1.0, link1 (slow) t0=1.2; caps 0.6*q
t0 = np.array([1.0 if i % 2 == 0 else 1.2 for i in range(m)])
alpha = np.full(m, 0.15)
beta = np.full(m, 4.0)
cap = np.full(m, 0.6 * CAPQ)

K = 1 << H
n = K
# paths: bitmask b -> stage s uses link 2*s + ((b>>s)&1)
indptr = np.arange(0, (K + 1) * H, H, dtype=np.int64)
indices = np.empty(K * H, dtype=np.int32)
for b in range(K):
    for s in range(H):
        indices[b * H + s] = 2 * s + ((b >> s) & 1)
p2od = np.zeros(n, dtype=np.int32)
d = np.array([Q])
x0 = np.full(n, Q / K)          # uniform cold start for every master

def w(name, arr, dtype):
    np.ascontiguousarray(arr, dtype=dtype).tofile(os.path.join(out, name))

w('B_indptr.i64', indptr, np.int64)
w('B_indices.i32', indices, np.int32)
w('p2od.i32', p2od, np.int32)
w('dvec.f64', d, np.float64)
w('x0.f64', x0, np.float64)
w('bpr_t0.f64', t0, np.float64)
w('bpr_alpha.f64', alpha, np.float64)
w('bpr_beta.f64', beta, np.float64)
w('bpr_cap.f64', cap, np.float64)
meta = dict(n=int(n), m=int(m), n_od=1, nnz=int(K * H), r=0,
            n_major=0, n_minor=0, svd_time_s=0.0, pool_file=tag)
json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'))
print(f'[{tag}] cpp_problem: K={K:,} paths, m={m} links, q={Q:,.0f}')

# ---- TNTP for tap-b: zones 1(origin),2(dest); interior nodes 3..H+1
# stage s (0-based): from node (1 if s==0 else s+2) to (2 if s==H-1 else s+3)
nnode = H + 1
with open(os.path.join(ROOT, f'{tag}_net.txt'), 'w') as f:
    f.write(f'<NUMBER OF ZONES> 2\n<NUMBER OF NODES> {nnode}\n'
            f'<FIRST THRU NODE> 3\n<NUMBER OF LINKS> {m}\n'
            '<END OF METADATA>\n\n'
            '~ 	tail	head	capacity	length	fftt	B	Power	speed	toll	type	\n')
    for s in range(H):
        a = 1 if s == 0 else s + 2
        b2 = 2 if s == H - 1 else s + 3
        for j in range(2):
            i = 2 * s + j
            f.write(f'\t{a}\t{b2}\t{cap[i]:.1f}\t1\t{t0[i]:.2f}\t'
                    f'{alpha[i]:.2f}\t{beta[i]:.0f}\t0\t0\t1\t;\n')
with open(os.path.join(ROOT, f'{tag}_trips.txt'), 'w') as f:
    f.write(f'<NUMBER OF ZONES> 2\n<TOTAL OD FLOW> {Q:.1f}\n'
            '<END OF METADATA>\n\n'
            f'Origin \t1\n\t2 :\t{Q:.1f};\n\nOrigin \t2\n')
with open(os.path.join(ROOT, f'{tag}_params.txt'), 'w') as f:
    f.write(f'<NETWORK FILE> {tag}_net.txt\n<TRIPS FILE> {tag}_trips.txt\n'
            '<CONVERGENCE GAP> 1e-6\n')
print(f'[{tag}] TNTP net/trips/params written')
