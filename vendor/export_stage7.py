#!/usr/bin/env python3
"""Export a GMNS (dataset_dir, pool) problem to the C++ solver's binary blobs — stage 7.

    python export_stage7.py <dataset_dir> <pool_file> <out_dir> [rank] [multi]

Differences from export_problem.py (which reads binary pool-master dumps via load_any):
  * loads through the CERTIFIED loader `compressed_assignment.load_problem`, so it accepts any
    GMNS path_pool.csv directly and supports `multi` (multi_path_only): keep only choice ODs
    and carry singleton demand as the constant background link flow P['v0'];
  * writes the additional blob v0.f64 whenever the background is nonzero. The C++ solver adds
    it to every from-scratch link-flow evaluation; the compressed blocks carry it inside
    v_base (build_compressed bakes it in), so Python and C++ price the identical objective.
  * rank=0 skips the compressed blocks entirely (full mode only).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compressed_assignment import load_problem, split_major_minor, build_compressed


def export(dataset_dir, pool_file, out_dir, rank=50, multi=False, tau=1.0,
           weighted=False):
    os.makedirs(out_dir, exist_ok=True)
    P = load_problem(dataset_dir, pool_file, multi_path_only=multi)
    major = split_major_minor(P, tau=tau)
    C = (build_compressed(P, major, rank,
                          weight_flows=(P['x0'] if weighted else None))
         if rank > 0 else None)

    B = P['B'].tocsr()

    def w(name, arr, dtype):
        np.ascontiguousarray(arr, dtype=dtype).tofile(os.path.join(out_dir, name))

    w('B_indptr.i64', B.indptr, np.int64)
    w('B_indices.i32', B.indices, np.int32)
    w('p2od.i32', P['p2od'], np.int32)
    w('dvec.f64', P['d'], np.float64)      # 'd.f64' would collide with 'D.f64' on Windows
    w('x0.f64', P['x0'], np.float64)
    w('major.u8', major.astype(np.uint8), np.uint8)
    bpr = P['bpr']
    w('bpr_t0.f64', bpr.t0, np.float64)
    w('bpr_alpha.f64', bpr.alpha, np.float64)
    w('bpr_beta.f64', bpr.beta, np.float64)
    w('bpr_cap.f64', bpr.cap, np.float64)
    v0 = np.asarray(P.get('v0', np.zeros(P['m']))).flatten()
    has_v0 = bool(np.abs(v0).max() > 0)
    if has_v0:
        w('v0.f64', v0, np.float64)
    if C is not None:
        w('U.f64', C['U'], np.float64)
        w('D.f64', C['D'], np.float64)
        w('M.f64', C['M'], np.float64)
        w('x0m.f64', C['x0m'], np.float64)
        w('d_eff.f64', C['d_eff'], np.float64)
        w('v_base.f64', C['v_base'], np.float64)   # includes v0 (build_compressed bakes it in)

    meta = {'n': int(P['n']), 'm': int(P['m']), 'n_od': int(P['n_od']),
            'nnz': int(B.nnz), 'r': int(C['r']) if C else 0,
            'n_major': int(major.sum()), 'n_minor': int((~major).sum()),
            'svd_time_s': round(C['svd_time'], 2) if C else 0.0,
            'pool_file': pool_file, 'multi_path_only': int(multi), 'has_v0': int(has_v0)}
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    export(sys.argv[1], sys.argv[2], sys.argv[3],
           rank=int(sys.argv[4]) if len(sys.argv) > 4 else 50,
           multi=bool(int(sys.argv[5])) if len(sys.argv) > 5 else False,
           tau=float(sys.argv[6]) if len(sys.argv) > 6 else 1.0,
           weighted=(len(sys.argv) > 7 and sys.argv[7] == 'weighted'))
