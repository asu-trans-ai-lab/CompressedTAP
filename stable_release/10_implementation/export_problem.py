#!/usr/bin/env python3
"""
Export a (pool, dataset) problem to flat binary blobs for the C++ solver.

Files written to <out_dir>/:
  meta.json                dims + file map
  B_indptr.i64, B_indices.i32   path-link incidence (binary; data implicitly 1)
  p2od.i32                 path -> OD index
  d.f64                    OD demand
  x0.f64                   nominal path flows (baseline store, zeros elsewhere)
  major.u8                 major mask
  bpr_t0.f64 bpr_alpha.f64 bpr_beta.f64 bpr_cap.f64     (per link)
  U.f64 (n_minor x r, row-major), D.f64 (m x r), M.f64 (n_od x r)
  x0m.f64, d_eff.f64, v_base.f64
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compressed_assignment import split_major_minor, build_compressed
from pathpool_io import load_any

def export(pool_file, out_dir, rank=50, dataset='03_chicago_sketch'):
    """rank=0 skips the compressed response blocks (U/D/M/...) entirely —
    use for pool-master-only problems (fw/gp/rsd/mlgp never read them; the
    minor-block SVD cost ~4h at ARC scale for nothing)."""
    data_dir = os.path.join(HERE, '..', 'data', dataset)
    os.makedirs(out_dir, exist_ok=True)
    P = load_any(data_dir, pool_file)      # binary pool; x0 = baseline only
    major = split_major_minor(P, tau=1.0)
    C = build_compressed(P, major, rank) if rank > 0 else None

    B = P['B'].tocsr()
    def w(name, arr, dtype):
        np.ascontiguousarray(arr, dtype=dtype).tofile(
            os.path.join(out_dir, name))

    w('B_indptr.i64', B.indptr, np.int64)
    w('B_indices.i32', B.indices, np.int32)
    w('p2od.i32', P['p2od'], np.int32)
    # NOTE: named dvec (not 'd') — Windows filenames are case-insensitive, so
    # 'd.f64' and the dense response block 'D.f64' would collide (found the
    # hard way: D overwrote the demand vector).
    w('dvec.f64', P['d'], np.float64)
    w('x0.f64', P['x0'], np.float64)
    w('major.u8', major.astype(np.uint8), np.uint8)
    bpr = P['bpr']
    w('bpr_t0.f64', bpr.t0, np.float64)
    w('bpr_alpha.f64', bpr.alpha, np.float64)
    w('bpr_beta.f64', bpr.beta, np.float64)
    w('bpr_cap.f64', bpr.cap, np.float64)
    if C is not None:
        w('U.f64', C['U'], np.float64)
        w('D.f64', C['D'], np.float64)
        w('M.f64', C['M'], np.float64)
        w('x0m.f64', C['x0m'], np.float64)
        w('d_eff.f64', C['d_eff'], np.float64)
        w('v_base.f64', C['v_base'], np.float64)

    meta = {'n': int(P['n']), 'm': int(P['m']), 'n_od': int(P['n_od']),
            'nnz': int(B.nnz), 'r': int(C['r']) if C else 0,
            'n_major': int(major.sum()), 'n_minor': int((~major).sum()),
            'svd_time_s': round(C['svd_time'], 2) if C else 0.0,
            'pool_file': pool_file}
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'E2_full_rich'
    dataset = sys.argv[2] if len(sys.argv) > 2 else '03_chicago_sketch'
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    tag = '' if dataset == '03_chicago_sketch' else '_regional'
    rtag = '' if rank == 50 else f'_r{rank}'
    export(f'path_pool_{which}.csv',
           os.path.join(HERE, '..', 'data', f'cpp_problem_{which}{tag}{rtag}'),
           rank=rank, dataset=dataset)
