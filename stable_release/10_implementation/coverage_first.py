"""Coverage-first pool construction (the mandatory precondition of every
experiment in this suite): given a network dataset and a source route
pool, produce a pool with C_OD = C_demand = 1 exactly, by appending one
free-flow shortest route per uncovered positive-demand OD pair
(one tree per origin, centroid-transit forbidden), and print the audit.
Richness (enrichment) is applied AFTER coverage, never instead of it.

usage: python coverage_first.py <dataset_dir> <src_pool.npz|-> <out_pool.npz> <n_zones>
       (src '-' builds a pure shortest-path pool from scratch)
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DD, SRC, OUT, N_ZONES = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

frm, to, lids, t0l = [], [], [], []
with open(os.path.join(DD, 'link.csv')) as f:
    for r in csv.DictReader(f):
        frm.append(int(r['from_node_id'])); to.append(int(r['to_node_id']))
        lids.append(int(r['link_id']))
        fftt = float(r.get('vdf_fftt') or 0.0)
        if fftt <= 0:
            ln = float(r.get('vdf_length_mi') or r.get('length') or 0.0)
            sp = float(r.get('vdf_free_speed_mph') or 30.0) or 30.0
            fftt = 60.0 * ln / sp
        t0l.append(max(fftt, 1e-4))
frm = np.array(frm); to = np.array(to); lids = np.array(lids)
t0l = np.array(t0l)
m = len(frm); n_node = int(max(frm.max(), to.max()))
linkid, rowof = {}, {}
for i in range(m):
    k = (frm[i], to[i])
    if k not in rowof or t0l[i] < t0l[rowof[k]]:
        linkid[k] = lids[i]; rowof[k] = i

q = {}
with open(os.path.join(DD, 'demand.csv')) as f:
    for r in csv.DictReader(f):
        o, d, v = int(r['o_zone_id']), int(r['d_zone_id']), float(r['volume'])
        if v > 0 and o != d:
            q[(o, d)] = q.get((o, d), 0.0) + v
tot = sum(q.values())
print(f'demand: {len(q):,} positive OD pairs, {tot:,.0f} veh')

if SRC != '-' and os.path.exists(os.path.join(DD, SRC)):
    z = np.load(os.path.join(DD, SRC), allow_pickle=True)
    indptr0 = z['indptr']; links0 = z['links']
    oz0 = z['o_zone']; dz0 = z['d_zone']; x00 = z['x0']; cost0 = z['cost']
    src0 = np.asarray(z['source'])
    names = z['source_names']
else:
    indptr0 = np.zeros(1, dtype=np.int64); links0 = np.zeros(0, dtype=np.int32)
    oz0 = np.zeros(0, dtype=np.int32); dz0 = np.zeros(0, dtype=np.int32)
    x00 = np.zeros(0); cost0 = np.zeros(0)
    src0 = np.zeros(0, dtype=np.int8); names = np.array(['sp_coverage'])
have = set(zip(oz0.tolist(), dz0.tolist()))
cq = sum(v for k, v in q.items() if k in have)
print(f'source pool: {len(indptr0)-1:,} paths; audit BEFORE: '
      f'C_OD={len([1 for k in q if k in have])/len(q):.4f} '
      f'C_demand={cq/tot:.4f}')

missing = [(o, d) for (o, d) in q if (o, d) not in have]
miss_by_o = {}
for (o, d) in missing:
    miss_by_o.setdefault(o, []).append(d)
base_rows = [i for i in range(m) if frm[i] > N_ZONES]
new_links, new_ptr, new_o, new_d, new_cost = [], [0], [], [], []
t0c = time.time()
for o in sorted(miss_by_o):
    rows = base_rows + [i for i in range(m) if frm[i] == o]
    g = csr_matrix((t0l[rows], (frm[rows] - 1, to[rows] - 1)),
                   shape=(n_node, n_node))
    dist, pred = dijkstra(g, indices=o - 1, return_predecessors=True)
    for d in miss_by_o[o]:
        if not np.isfinite(dist[d - 1]): continue
        seq = []; node = d - 1; bad = False
        while node != o - 1:
            p = pred[node]
            if p < 0: bad = True; break
            seq.append(linkid[(p + 1, node + 1)]); node = p
        if bad: continue
        seq.reverse()
        new_links.extend(seq); new_ptr.append(len(new_links))
        new_o.append(o); new_d.append(d); new_cost.append(float(dist[d - 1]))
n_new = len(new_o)
print(f'SP completion: {n_new:,} routes '
      f'({time.time()-t0c:.0f}s); unreachable: {len(missing)-n_new}')

indptr = np.concatenate([indptr0, indptr0[-1] +
                         np.array(new_ptr[1:], dtype=np.int64)]) \
         if n_new else indptr0
links = np.concatenate([links0, np.array(new_links, dtype=np.int32)])
oz = np.concatenate([oz0, np.array(new_o, dtype=np.int32)])
dz = np.concatenate([dz0, np.array(new_d, dtype=np.int32)])
x0 = np.concatenate([x00, np.zeros(n_new)])
cost = np.concatenate([cost0, np.array(new_cost)])
srcv = np.concatenate([src0, np.full(n_new, 9, dtype=np.int8)])
np.savez_compressed(os.path.join(DD, OUT),
                    indptr=indptr, links=links, o_zone=oz, d_zone=dz,
                    x0=x0, cost=cost, source=srcv, source_names=names)
cov = set(zip(oz.tolist(), dz.tolist()))
cq2 = sum(v for k, v in q.items() if k in cov)
cod = len([1 for k in q if k in cov]) / len(q)
print(f'audit AFTER: {len(indptr)-1:,} paths, C_OD={cod:.4f} '
      f'C_demand={cq2/tot:.4f} K/OD={(len(indptr)-1)/len(q):.2f}')
assert abs(cq2/tot - 1.0) < 1e-9 and abs(cod - 1.0) < 1e-9, 'COVERAGE INCOMPLETE'
print('COVERAGE-FIRST-OK')
