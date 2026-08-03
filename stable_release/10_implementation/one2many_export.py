"""Tier-0b generator: ONE origin, D destinations over a shared layered
trunk (H stages x 2 parallel links) followed by per-destination layered
branches (H2 stages x 2 links). OD pairs couple through trunk congestion
(origin-block structure, family-A style). Pool enumerates ALL
2^(H+H2) routes per OD, so pool and full-network solvers face the same
problem. Emits cpp_problem_<tag>/ + TNTP for tap-b.

usage: python one2many_export.py <H_trunk> <H2_branch> <D> <q_per_dest> <out_root> <tag>
"""
import json
import os
import sys

import numpy as np

H, H2, D, Q, ROOT, TAG = (int(sys.argv[1]), int(sys.argv[2]),
                          int(sys.argv[3]), float(sys.argv[4]),
                          sys.argv[5], sys.argv[6])
CAPQ = float(sys.argv[7]) if len(sys.argv) > 7 else Q
out = os.path.join(ROOT, f'cpp_problem_{TAG}')
os.makedirs(out, exist_ok=True)

m_trunk = 2 * H
m = m_trunk + D * 2 * H2
t0 = np.empty(m); cap = np.empty(m)
for s in range(H):                       # trunk: carries D*Q at even split
    t0[2*s] = 1.0; t0[2*s+1] = 1.2
    cap[2*s] = cap[2*s+1] = 0.55 * D * CAPQ
for dd in range(D):
    for s in range(H2):
        i = m_trunk + dd*2*H2 + 2*s
        t0[i] = 1.0; t0[i+1] = 1.2
        cap[i] = cap[i+1] = 0.55 * CAPQ
alpha = np.full(m, 0.15); beta = np.full(m, 4.0)

Kpo = 1 << (H + H2)                      # routes per OD
n = D * Kpo
L = H + H2
indptr = np.arange(0, (n + 1) * L, L, dtype=np.int64)
indices = np.empty(n * L, dtype=np.int32)
p2od = np.empty(n, dtype=np.int32)
p = 0
for dd in range(D):
    for b in range(Kpo):
        for s in range(H):
            indices[p*L + s] = 2*s + ((b >> s) & 1)
        for s in range(H2):
            indices[p*L + H + s] = m_trunk + dd*2*H2 + 2*s + ((b >> (H+s)) & 1)
        p2od[p] = dd
        p += 1
d = np.full(D, Q)
x0 = np.full(n, Q / Kpo)

def w(name, arr, dtype):
    np.ascontiguousarray(arr, dtype=dtype).tofile(os.path.join(out, name))
w('B_indptr.i64', indptr, np.int64); w('B_indices.i32', indices, np.int32)
w('p2od.i32', p2od, np.int32); w('dvec.f64', d, np.float64)
w('x0.f64', x0, np.float64); w('bpr_t0.f64', t0, np.float64)
w('bpr_alpha.f64', alpha, np.float64); w('bpr_beta.f64', beta, np.float64)
w('bpr_cap.f64', cap, np.float64)
json.dump(dict(n=int(n), m=int(m), n_od=int(D), nnz=int(n*L), r=0,
               n_major=0, n_minor=0, svd_time_s=0.0, pool_file=TAG),
          open(os.path.join(out, 'meta.json'), 'w'))
print(f'[{TAG}] one2many: D={D} dests, {Kpo:,} routes/OD, {n:,} total, '
      f'm={m} links, q={Q:,.0f}/dest (trunk v/c at even split: '
      f'{D*Q/2/cap[0]:.2f})')

# TNTP: zones 1..D+1 (1=origin, 2..D+1=dests); interior D+2..
# trunk nodes: n_or=1 -> t1..tH (interior); branch dd: tH -> b_{dd,1..H2-1} -> zone dd+2
nid = D + 2
trunk_nodes = []
for s in range(H):
    trunk_nodes.append(nid); nid += 1
branch_nodes = {}
for dd in range(D):
    branch_nodes[dd] = []
    for s in range(H2 - 1):
        branch_nodes[dd].append(nid); nid += 1
nnode = nid - 1
lines = []
for s in range(H):
    a = 1 if s == 0 else trunk_nodes[s-1]
    b2 = trunk_nodes[s]
    for j in range(2):
        i = 2*s + j
        lines.append((a, b2, cap[i], t0[i]))
for dd in range(D):
    for s in range(H2):
        a = trunk_nodes[H-1] if s == 0 else branch_nodes[dd][s-1]
        b2 = (dd + 2) if s == H2 - 1 else branch_nodes[dd][s]
        for j in range(2):
            i = m_trunk + dd*2*H2 + 2*s + j
            lines.append((a, b2, cap[i], t0[i]))
with open(os.path.join(ROOT, f'{TAG}_net.txt'), 'w') as f:
    f.write(f'<NUMBER OF ZONES> {D+1}\n<NUMBER OF NODES> {nnode}\n'
            f'<FIRST THRU NODE> {D+2}\n<NUMBER OF LINKS> {m}\n'
            '<END OF METADATA>\n\n~ tail head cap len fftt B pow spd toll type\n')
    for (a, b2, c2, tt) in lines:
        f.write(f'\t{a}\t{b2}\t{c2:.1f}\t1\t{tt:.2f}\t0.15\t4\t0\t0\t1\t;\n')
with open(os.path.join(ROOT, f'{TAG}_trips.txt'), 'w') as f:
    f.write(f'<NUMBER OF ZONES> {D+1}\n<TOTAL OD FLOW> {D*Q:.1f}\n'
            '<END OF METADATA>\n\nOrigin \t1\n')
    for dd in range(D):
        f.write(f'\t{dd+2} :\t{Q:.1f};\n')
    for dd in range(D):
        f.write(f'\nOrigin \t{dd+2}\n')
with open(os.path.join(ROOT, f'{TAG}_params.txt'), 'w') as f:
    f.write(f'<NETWORK FILE> {TAG}_net.txt\n<TRIPS FILE> {TAG}_trips.txt\n'
            '<CONVERGENCE GAP> 1e-6\n')
print(f'[{TAG}] TNTP written ({nnode} nodes)')
