"""Gate: do C++ and Python produce consistent v3 metrics on the SAME instance?

Runs the Python solve_compressed on the identical (P, C), reads the C++ raw-x dump, and
computes the full v3 metric set (delta_F, Gap_F, R2) on BOTH terminal iterates through the
one shared v3_metrics module, against a common reference. If the two solutions agree in
objective and the metrics agree, C++ can replace the Python solve in Panel A without
changing what any table reports.

    python validate_cpp_metrics.py <export_dir> <dataset_dir> <pool> <multi> <tau> <rank>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CERT = HERE.parents[3] / "source" / "updated_TAPLite" / "python"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CERT))

import v3_metrics as M
import compressed_assignment as ca
from run_table2_consistency import op_gp_full


def main():
    export_dir, dsdir, pool, multi, tau, rank = sys.argv[1:7]
    multi = bool(int(multi)); tau = float(tau); rank = int(rank)

    P = ca.load_problem(dsdir, pool, multi_path_only=multi)
    sl = M.od_slice_index(P)
    major = ca.split_major_minor(P, tau=tau)
    C = ca.build_compressed(P, major, rank)
    print(f"instance: n={P['n']} od={P['n_od']} majors={int(major.sum())} r={C['r']}")

    # common reference: a well-converged full-path GP (small tol) so Gap_F is meaningful
    print("reference: full-path GP ...", flush=True)
    xr, tr, itr, gr = op_gp_full(P, 1e-6, max_seconds=600)
    xref = M.convert_euclid(xr, P["d"], P["p2od"])
    vref = M.link_flow(P, xref); fref = float(P["bpr"].beckmann(vref))
    print(f"  f_ref={fref:,.4f} cert={gr:.3e} t={tr:.1f}s")

    # Python compressed solve on the SAME (P, C)
    t = time.perf_counter()
    solp = ca.solve_compressed(P, C, regime="hard", tol=1e-4, max_outer=30)
    tp = time.perf_counter() - t
    mp = M.evaluate(P, solp["x_raw"], fref, vref, "euclid", sl)
    print(f"\nPython solve_compressed: t={tp:.1f}s obj={float(P['bpr'].beckmann(solp['v'])):,.4f}")
    print(f"  delta_F={mp['delta_F_pct']:.4f}%  Gap_F={mp['Gap_F_pct']:+.5f}%  "
          f"R2={mp['link_R2']:.6f}")

    # C++ raw-x readback on the SAME instance
    xraw_path = Path(export_dir) / "x_raw.f64"
    if not xraw_path.exists():
        print(f"\nC++ raw-x not found at {xraw_path}; run compressed_solver with "
              f"VDUMP_XRAW set first.")
        return 1
    xc = np.fromfile(xraw_path, dtype=np.float64)
    print(f"\nC++ raw-x: {xc.size} entries (expected {P['n']})")
    if xc.size != P["n"]:
        print("  SIZE MISMATCH -- instances differ, cannot compare"); return 1
    mc = M.evaluate(P, xc, fref, vref, "euclid", sl)
    print(f"  delta_F={mc['delta_F_pct']:.4f}%  Gap_F={mc['Gap_F_pct']:+.5f}%  "
          f"R2={mc['link_R2']:.6f}")

    print("\n=== consistency (C++ vs Python, same instance, same metrics module) ===")
    dG = abs(mc["Gap_F_pct"] - mp["Gap_F_pct"])
    dD = abs(mc["delta_F_pct"] - mp["delta_F_pct"])
    dR = abs(mc["link_R2"] - mp["link_R2"])
    print(f"  |Delta Gap_F| = {dG:.4f} pp   |Delta delta_F| = {dD:.4f} pp   "
          f"|Delta R2| = {dR:.6f}")
    # both are truncated solves of the same convex problem; expect close, not identical
    ok = dR < 5e-3 and dG < 0.5
    print(f"  VERDICT: {'CONSISTENT' if ok else 'REVIEW'} "
          f"(link R2 within 5e-3 and Gap_F within 0.5pp)")
    print(f"\n  speedup this instance: Python {tp:.0f}s  (C++ measured separately in its log)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
