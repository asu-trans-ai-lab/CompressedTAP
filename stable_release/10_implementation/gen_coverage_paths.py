"""Coverage completion for the Regional pool (user protocol 2026-07-12):
one free-flow shortest-path tree per origin, first_thru_node enforced
(centroid nodes are never intermediate), one path for every
positive-demand OD missing from the pool; merge into a new npz pool.

output: data/04_chicago_regional/path_pool_E0_cov.npz
"""
import csv
import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(HERE, '..', 'data', '04_chicago_regional')
N_ZONES = 1790                      # TNTP: nodes <= N_ZONES are centroids

# ---- network
frm, to, t0l, lids = [], [], [], []
with open(os.path.join(DD, 'link.csv')) as f:
    for r in csv.DictReader(f):
        frm.append(int(r['from_node_id']))
        to.append(int(r['to_node_id']))
        lids.append(int(r['link_id']))
        fftt = float(r.get('vdf_fftt') or 0.0)
        if fftt <= 0:               # connectors: derive from length/speed
            ln = float(r.get('vdf_length_mi') or r.get('length') or 0.0)
            sp = float(r.get('vdf_free_speed_mph') or 30.0) or 30.0
            fftt = 60.0 * ln / sp
        t0l.append(max(fftt, 1e-4))
frm = np.array(frm); to = np.array(to); t0l = np.array(t0l)
lids = np.array(lids)
m = len(frm)
n_node = int(max(frm.max(), to.max()))
linkid = {}
rowof = {}
for i in range(m):                  # keep cheapest parallel link
    k = (frm[i], to[i])
    if k not in linkid or t0l[i] < t0l[rowof[k]]:
        linkid[k] = lids[i]         # pool stores link_id (1-based), not row
        rowof[k] = i
print(f'network: {n_node:,} nodes, {m:,} links')

# ---- demand
q = {}
with open(os.path.join(DD, 'demand.csv')) as f:
    for r in csv.DictReader(f):
        o, d, v = int(r['o_zone_id']), int(r['d_zone_id']), float(r['volume'])
        if v > 0 and o != d:
            q[(o, d)] = q.get((o, d), 0.0) + v
dest_by_o = {}
for (o, d) in q:
    dest_by_o.setdefault(o, []).append(d)
print(f'demand: {len(q):,} positive ODs, {sum(q.values()):,.0f} veh')

# ---- existing pool
z = np.load(os.path.join(DD, 'path_pool_E0_baseline.npz'), allow_pickle=True)
indptr0 = z['indptr']; links0 = z['links']
oz0 = z['o_zone']; dz0 = z['d_zone']; x00 = z['x0']; cost0 = z['cost']
have = set(zip(oz0.tolist(), dz0.tolist()))
missing = [(o, d) for (o, d) in q if (o, d) not in have]
print(f'pool: {len(indptr0)-1:,} paths, {len(have):,} covered ODs; '
      f'missing {len(missing):,} ODs '
      f'({sum(q[k] for k in missing):,.0f} veh)')

# ---- SP trees: graph WITHOUT arcs leaving centroids (except per-origin)
base_rows = [i for i in range(m) if frm[i] > N_ZONES]
miss_by_o = {}
for (o, d) in missing:
    miss_by_o.setdefault(o, []).append(d)

new_links, new_ptr, new_o, new_d, new_cost = [], [0], [], [], []
t0c = time.time()
done = 0
for o in sorted(miss_by_o):
    rows = base_rows + [i for i in range(m) if frm[i] == o]
    g = csr_matrix((t0l[rows], (frm[rows] - 1, to[rows] - 1)),
                   shape=(n_node, n_node))
    dist, pred = dijkstra(g, indices=o - 1, return_predecessors=True)
    for d in miss_by_o[o]:
        if not np.isfinite(dist[d - 1]):
            continue                # disconnected OD: report below
        seq = []
        node = d - 1
        while node != o - 1:
            p = pred[node]
            if p < 0: seq = None; break
            seq.append(linkid[(p + 1, node + 1)])
            node = p
        if seq is None:
            continue
        seq.reverse()
        new_links.extend(seq)
        new_ptr.append(len(new_links))
        new_o.append(o); new_d.append(d)
        new_cost.append(float(dist[d - 1]))
    done += 1
    if done % 200 == 0:
        print(f'  {done}/{len(miss_by_o)} origins, '
              f'{len(new_o):,} paths, {time.time()-t0c:.0f}s', flush=True)
n_new = len(new_o)
unreach = len(missing) - n_new
print(f'SP coverage: {n_new:,} new paths; unreachable ODs: {unreach}')

# ---- merge and save
indptr = np.concatenate([indptr0,
                         indptr0[-1] + np.array(new_ptr[1:], dtype=indptr0.dtype)])
links = np.concatenate([links0, np.array(new_links, dtype=links0.dtype)])
oz = np.concatenate([oz0, np.array(new_o, dtype=oz0.dtype)])
dz = np.concatenate([dz0, np.array(new_d, dtype=dz0.dtype)])
x0 = np.concatenate([x00, np.zeros(n_new, dtype=x00.dtype)])
cost = np.concatenate([cost0, np.array(new_cost, dtype=cost0.dtype)])
src = z['source'] if 'source' in z else np.zeros(len(x00), dtype=np.int8)
src = np.concatenate([np.asarray(src),
                      np.full(n_new, 9, dtype=np.asarray(src).dtype)])
np.savez_compressed(os.path.join(DD, 'path_pool_E0_cov.npz'),
                    indptr=indptr, links=links, o_zone=oz, d_zone=dz,
                    x0=x0, cost=cost, source=src,
                    source_names=np.array(['..'] * 9 + ['sp_coverage']))
cov = set(zip(oz.tolist(), dz.tolist()))
cq = sum(v for k, v in q.items() if k in cov)
print(f'MERGED: {len(indptr)-1:,} paths, {len(cov):,} ODs; '
      f'C_OD={len([1 for k in q if k in cov])/len(q):.4f} '
      f'C_demand={cq/sum(q.values()):.4f}')
print('COVERAGE-DONE')
