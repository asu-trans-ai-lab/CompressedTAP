"""Enrichment round for the coverage-completed Regional pool (protocol
step 6): one SP tree per origin at CONGESTED times taken from the latest
latent-GP solution, appended as alternative routes for the single-route
(coverage) ODs; re-export rank-0. Usage: python enrich_coverage.py <round>
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_problem import export

RND = int(sys.argv[1])
DD = os.path.join(HERE, '..', 'data', '04_chicago_regional')
SRC_NPZ = ('path_pool_E0_cov.npz' if RND == 1
           else f'path_pool_E0_cov_e{RND-1}.npz')
OUT_NPZ = f'path_pool_E0_cov_e{RND}.npz'
VOL = os.path.join(HERE, '..', 'kernel',
                   'v_cov_latgp.f64' if RND == 1 else f'v_cove{RND-1}_latgp.f64')
N_ZONES = 1790

frm, to, lids, t0l, alp, bet, cap = [], [], [], [], [], [], []
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
        alp.append(float(r.get('vdf_alpha') or 0.15))
        bet.append(float(r.get('vdf_beta') or 4.0))
        cap.append(max(float(r.get('capacity') or 1000.0), 1.0))
frm = np.array(frm); to = np.array(to); lids = np.array(lids)
t0l = np.array(t0l); alp = np.array(alp); bet = np.array(bet); cap = np.array(cap)
m = len(frm); n_node = int(max(frm.max(), to.max()))
v = np.fromfile(VOL)
tcong = t0l * (1.0 + alp * np.power(np.maximum(v, 0.0) / cap, bet))
print(f'[e{RND}] congested times from {os.path.basename(VOL)}: '
      f'mean t/t0 = {float(np.mean(tcong/t0l)):.2f}', flush=True)

linkid, rowof = {}, {}
for i in range(m):
    k = (frm[i], to[i])
    if k not in rowof or tcong[i] < tcong[rowof[k]]:
        linkid[k] = lids[i]; rowof[k] = i

z = np.load(os.path.join(DD, SRC_NPZ), allow_pickle=True)
indptr0 = z['indptr']; links0 = z['links']
oz0 = z['o_zone']; dz0 = z['d_zone']; x00 = z['x0']; cost0 = z['cost']
src0 = np.asarray(z['source'])
n0 = len(indptr0) - 1
Kw = {}
for i in range(n0):
    Kw[(int(oz0[i]), int(dz0[i]))] = Kw.get((int(oz0[i]), int(dz0[i])), 0) + 1
targets = {}
for w, k in Kw.items():
    if k == 1:
        targets.setdefault(w[0], []).append(w[1])
print(f'[e{RND}] single-route ODs to enrich: '
      f'{sum(len(v2) for v2 in targets.values()):,}', flush=True)

seen = set()
for i in range(n0):
    seen.add((int(oz0[i]), int(dz0[i]),
              tuple(links0[indptr0[i]:indptr0[i+1]].tolist())))
base_rows = [i for i in range(m) if frm[i] > N_ZONES]
new_links, new_ptr, new_o, new_d, new_cost = [], [0], [], [], []
t0c = time.time(); done = 0
for o in sorted(targets):
    rows = base_rows + [i for i in range(m) if frm[i] == o]
    g = csr_matrix((tcong[rows], (frm[rows] - 1, to[rows] - 1)),
                   shape=(n_node, n_node))
    dist, pred = dijkstra(g, indices=o - 1, return_predecessors=True)
    for d in targets[o]:
        if not np.isfinite(dist[d - 1]): continue
        seq = []; node = d - 1; bad = False
        while node != o - 1:
            p = pred[node]
            if p < 0: bad = True; break
            seq.append(linkid[(p + 1, node + 1)]); node = p
        if bad: continue
        seq.reverse()
        key = (o, d, tuple(seq))
        if key in seen: continue
        seen.add(key)
        new_links.extend(seq); new_ptr.append(len(new_links))
        new_o.append(o); new_d.append(d); new_cost.append(float(dist[d - 1]))
    done += 1
    if done % 300 == 0:
        print(f'  {done}/{len(targets)} origins, {len(new_o):,} new paths, '
              f'{time.time()-t0c:.0f}s', flush=True)
n_new = len(new_o)
print(f'[e{RND}] appended {n_new:,} congested-SP alternatives', flush=True)

indptr = np.concatenate([indptr0, indptr0[-1] +
                         np.array(new_ptr[1:], dtype=indptr0.dtype)])
links = np.concatenate([links0, np.array(new_links, dtype=links0.dtype)])
oz = np.concatenate([oz0, np.array(new_o, dtype=oz0.dtype)])
dz = np.concatenate([dz0, np.array(new_d, dtype=dz0.dtype)])
x0 = np.concatenate([x00, np.zeros(n_new, dtype=x00.dtype)])
cost = np.concatenate([cost0, np.array(new_cost, dtype=cost0.dtype)])
srcv = np.concatenate([src0, np.full(n_new, 9, dtype=src0.dtype)])
np.savez_compressed(os.path.join(DD, OUT_NPZ),
                    indptr=indptr, links=links, o_zone=oz, d_zone=dz,
                    x0=x0, cost=cost, source=srcv,
                    source_names=z['source_names'])
out = os.path.join(HERE, '..', 'data', f'cpp_problem_E0_cove{RND}_regional')
export(OUT_NPZ, out, rank=0, dataset='04_chicago_regional')
print(f'ENRICH-{RND}-DONE ({time.time()-t0c:.0f}s)', flush=True)
