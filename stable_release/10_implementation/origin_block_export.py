"""Tier-0c: one origin -> many destinations on a REAL network, extracted
from a coverage-complete pool (so the block inherits C=1 for its ODs).
Picks the maximum-demand origin unless one is given. Emits
cpp_problem_<tag>/ (full link set, subset paths) + a one-origin TNTP
trips file for tap-b.

usage: python origin_block_export.py <dataset_dir> <pool.npz> <out_root> <tag> [origin]
"""
import csv
import json
import os
import sys

import numpy as np

DD, POOL, ROOT, TAG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ORIGIN = int(sys.argv[5]) if len(sys.argv) > 5 else 0

# demand
q = {}
for r in csv.DictReader(open(os.path.join(DD, 'demand.csv'))):
    o, d, v = int(r['o_zone_id']), int(r['d_zone_id']), float(r['volume'])
    if v > 0 and o != d:
        q[(o, d)] = q.get((o, d), 0.0) + v
if ORIGIN == 0:
    per_o = {}
    for (o, d), v in q.items():
        per_o[o] = per_o.get(o, 0.0) + v
    ORIGIN = max(per_o, key=per_o.get)
dests = sorted(d for (o, d) in q if o == ORIGIN)
qsub = {d: q[(ORIGIN, d)] for d in dests}
print(f'[{TAG}] origin {ORIGIN}: {len(dests)} destinations, '
      f'{sum(qsub.values()):,.0f} veh')

# links (for m + bpr params; pool link ids are 1-based rows)
t0l, alp, bet, cap = [], [], [], []
for r in csv.DictReader(open(os.path.join(DD, 'link.csv'))):
    fftt = float(r.get('vdf_fftt') or 0.0)
    if fftt <= 0:
        ln = float(r.get('vdf_length_mi') or r.get('length') or 0.0)
        sp = float(r.get('vdf_free_speed_mph') or 30.0) or 30.0
        fftt = 60.0 * ln / sp
    t0l.append(max(fftt, 1e-4))
    alp.append(float(r.get('vdf_alpha') or 0.15))
    bet.append(float(r.get('vdf_beta') or 4.0))
    cap.append(max(float(r.get('capacity') or 1000.0), 1.0))
m = len(t0l)

z = np.load(os.path.join(DD, POOL), allow_pickle=True)
indptr = z['indptr']; links = z['links']
oz = np.asarray(z['o_zone']); dz = np.asarray(z['d_zone'])
x0all = z['x0']
sel = np.where(oz == ORIGIN)[0]
dset = {d: i for i, d in enumerate(dests)}
keep = [p for p in sel if int(dz[p]) in dset]
print(f'[{TAG}] pool paths for origin: {len(keep):,} '
      f'(K/OD {len(keep)/len(dests):.2f})')
covered = set(int(dz[p]) for p in keep)
assert covered == set(dests), 'origin block not coverage-complete'

out = os.path.join(ROOT, f'cpp_problem_{TAG}')
os.makedirs(out, exist_ok=True)
nptr = [0]; nidx = []; p2od = []; x0 = []
for p in keep:
    seg = links[indptr[p]:indptr[p + 1]]
    nidx.extend((np.asarray(seg) - 1).tolist())   # 1-based id -> 0-based row
    nptr.append(len(nidx))
    p2od.append(dset[int(dz[p])])
    x0.append(float(x0all[p]))
d = np.array([qsub[d2] for d2 in dests])

def w(name, arr, dtype):
    np.ascontiguousarray(arr, dtype=dtype).tofile(os.path.join(out, name))
w('B_indptr.i64', np.array(nptr), np.int64)
w('B_indices.i32', np.array(nidx), np.int32)
w('p2od.i32', np.array(p2od), np.int32)
w('dvec.f64', d, np.float64)
w('x0.f64', np.array(x0), np.float64)
w('bpr_t0.f64', np.array(t0l), np.float64)
w('bpr_alpha.f64', np.array(alp), np.float64)
w('bpr_beta.f64', np.array(bet), np.float64)
w('bpr_cap.f64', np.array(cap), np.float64)
json.dump(dict(n=len(keep), m=m, n_od=len(dests), nnz=len(nidx), r=0,
               n_major=0, n_minor=0, svd_time_s=0.0,
               pool_file=f'{TAG}(origin {ORIGIN})'),
          open(os.path.join(out, 'meta.json'), 'w'))

# one-origin TNTP trips (net file: reuse the network's TNTP net)
with open(os.path.join(ROOT, f'{TAG}_trips.txt'), 'w') as f:
    nz = max(max(o for o, _ in q), max(d2 for _, d2 in q))
    f.write(f'<NUMBER OF ZONES> {nz}\n'
            f'<TOTAL OD FLOW> {sum(qsub.values()):.1f}\n'
            '<END OF METADATA>\n\n')
    for o in range(1, nz + 1):
        f.write(f'Origin \t{o}\n')
        if o == ORIGIN:
            for d2 in dests:
                f.write(f'\t{d2} :\t{qsub[d2]:.2f};\n')
        f.write('\n')
print(f'[{TAG}] exported (origin {ORIGIN}); trips file written')
