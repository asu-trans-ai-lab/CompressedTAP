"""Targeted re-run of the single Regional (E0) tau=1.03 Panel A row.

The full driver died mid-row when the session restarted; tau=0/0.23/0.46 are already in
results/table4A_regional_cpp.csv. This solves ONLY the cached tau=1.03 export and appends the
row, reusing the cached tau=0 reference raw flows (SCRATCH/regional/tau0/x_raw.f64) so the
Gap_F reference is byte-identical to the other three rows. Same solver settings as the driver
(hard, max_outer=40, rank=50, max_inner=300).

    python run_regional_tau103.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CERTPY = ROOT / "vendor"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CERTPY))

import v3_metrics as M
import compressed_assignment as ca
from run_table4A_cpp import NETS, SCRATCH, cpp_solve

TAU = 1.03
CSV = HERE.parent / "results" / "table4A_regional_cpp.csv"


def main():
    cfg = NETS["regional"]
    print("[tau1.03] loading regional (E0) for metrics", flush=True)
    P = ca.load_problem(str(cfg["dir"]), cfg["pool"], multi_path_only=bool(cfg["multi"]))
    sl = M.od_slice_index(P)
    n = P["n"]
    print(f"  n={n:,} od={P['n_od']:,}", flush=True)

    # reference = cached tau=0 full solve
    ref = SCRATCH / "regional" / "tau0" / "x_raw.f64"
    x0raw = np.fromfile(ref, dtype=np.float64)
    assert x0raw.size == n, f"reference size {x0raw.size} != n {n}"
    x0f = M.convert_euclid(x0raw, P["d"], P["p2od"])
    v_ref = M.link_flow(P, x0f); f_ref = float(P["bpr"].beckmann(v_ref))
    print(f"  f_ref(tau0)={f_ref:,.4f}", flush=True)

    expd = SCRATCH / "regional" / f"tau{TAU}"
    import json
    meta = json.load(open(expd / "meta.json"))
    s_major = int(meta["n_major"]); r_used = int(meta["r"])
    reduction = 100.0 * (n - s_major - r_used) / n
    print(f"[tau1.03] solving cached export  n_minor={meta['n_minor']:,} "
          f"reduction={reduction:.1f}%", flush=True)
    x_raw, secs, info, _ = cpp_solve(expd, "hard", 40, 50, 300)
    m = M.evaluate_both(P, x_raw, f_ref, v_ref, sl)
    m.update(dict(net="regional", tau=TAU, n_paths=n, n_od=P["n_od"], majors=s_major,
                  rank=r_used, reduction_pct=reduction,
                  raw_obj_diff_pct=M.raw_objective_diff_pct(P, x_raw, f_ref),
                  inner_iters=info.get("inner_iters"), cpu_s=secs,
                  cpp_obj_feasible=info.get("obj_feasible"),
                  cpp_cons=info.get("cons"), cpp_pool_gap=info.get("pool_gap"),
                  solver="cpp", ref_source="cpp-tau0-full"))
    print(f"  red={reduction:5.1f}%  dF={m['delta_F_pct']:.4f}%  "
          f"GapF={m['Gap_F_pct']:+.5f}%  R2={m['link_R2']:.4f}  cpu={secs:.1f}s", flush=True)

    # append to CSV (preserve existing 3 rows, union of columns)
    rows = list(csv.DictReader(CSV.open())) if CSV.exists() else []
    rows = [r for r in rows if r.get("tau") != str(TAU)]  # drop any prior tau1.03
    rows.append({k: m.get(k, "") for k in m})
    keys = sorted({k for r in rows for k in r})
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[tau1.03] appended -> {CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
