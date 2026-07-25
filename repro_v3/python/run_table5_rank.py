"""Table 5 (tab:impact-rank) via the certified C++ solver.

Rank sensitivity at a FIXED threshold: Regional (tau=1.03) and Philadelphia (tau=5.33),
r in {50,100,150,200}. Reuses run_table4A_cpp's export/solve/metrics helpers so every number
means exactly what Panel A's numbers mean (same instances, same shared v3_metrics module).

Fills the Feasible-Obj-Gap column; CPU / iterations / R^2 are re-timed and re-emitted per
decision 2. Reference (decision A / finalize_panelA): the best feasible objective found by any
run on that network -- here the tau=0 full solve and the four rank solves -- so Gap_F >= 0 by
construction and the best row is 0. Networks use the SAME submitted instances as Panel A:
Regional = E0 baseline pool (author decision 2026-07-24), Philadelphia = V2 pool.

    python run_table5_rank.py --net regional      # tau=1.03
    python run_table5_rank.py --net philadelphia  # tau=5.33
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v3_metrics as M
import compressed_assignment as ca
from run_table4A_cpp import NETS, SCRATCH, export, cpp_solve

# fixed threshold per network (the most aggressive Panel A row, matching the manuscript)
TAU_FIXED = {"regional": 1.03, "philadelphia": 5.33}
RANKS = [50, 100, 150, 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=list(TAU_FIXED))
    ap.add_argument("--max-outer", type=int, default=40)
    ap.add_argument("--max-inner", type=int, default=300)
    ap.add_argument("--ref-outer", type=int, default=40)
    ap.add_argument("--reuse-ref", action="store_true",
                    help="reuse Panel A's tau=0 reference raw flows from SCRATCH/<net>/tau0/"
                         "x_raw.f64 instead of re-solving the expensive full reference")
    a = ap.parse_args()
    cfg = NETS[a.net]
    tau = TAU_FIXED[a.net]

    print(f"[T5-rank] loading {a.net} for metrics", flush=True)
    P = ca.load_problem(str(cfg["dir"]), cfg["pool"], multi_path_only=bool(cfg["multi"]))
    sl = M.od_slice_index(P)
    n = P["n"]
    print(f"  n={n:,} od={P['n_od']:,} tau={tau}", flush=True)

    # reference = tau=0 full solve (same construction as Panel A). --reuse-ref loads the
    # identical reference Panel A already dumped, avoiding a second expensive full solve.
    exp0 = SCRATCH / a.net / "tau0"
    ref_dump = exp0 / "x_raw.f64"
    t0 = 0.0
    if a.reuse_ref and ref_dump.exists():
        print(f"[T5-rank] reusing Panel A reference {ref_dump}", flush=True)
        x0raw = np.fromfile(ref_dump, dtype=np.float64)
        if x0raw.size != n:
            print(f"  reference size {x0raw.size} != n {n}; re-solving instead", flush=True)
            a.reuse_ref = False
    if not (a.reuse_ref and ref_dump.exists() and x0raw.size == n):
        print("[T5-rank] reference = C++ tau=0 full solve", flush=True)
        export(cfg["dir"], cfg["pool"], cfg["multi"], 0.0, 0, exp0)
        x0raw, t0, info0, _ = cpp_solve(exp0, "full", a.ref_outer, 0, a.max_inner)
    x0f = M.convert_euclid(x0raw, P["d"], P["p2od"])
    v_ref0 = M.link_flow(P, x0f); f_ref0 = float(P["bpr"].beckmann(v_ref0))
    print(f"  f_ref(tau0)={f_ref0:,.4f} t={t0:.1f}s", flush=True)

    out = HERE.parent / "results" / f"table5_rank_{a.net}_cpp.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def flush():
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    for r in RANKS:
        print(f"[T5-rank] r={r}", flush=True)
        expd = SCRATCH / a.net / f"tau{tau}_r{r}"
        meta = export(cfg["dir"], cfg["pool"], cfg["multi"], tau, r, expd)
        s_major = int(meta["n_major"]); r_used = int(meta["r"])
        reduction = 100.0 * (n - s_major - r_used) / n
        x_raw, secs, info, _ = cpp_solve(expd, "hard", a.max_outer, r, a.max_inner)
        m = M.evaluate_both(P, x_raw, f_ref0, v_ref0, sl)
        m.update(dict(net=a.net, tau=tau, rank=r_used, majors=s_major, n_paths=n,
                      reduction_pct=reduction,
                      raw_obj_diff_pct=M.raw_objective_diff_pct(P, x_raw, f_ref0),
                      inner_iters=info.get("inner_iters"), cpu_s=secs,
                      cpp_obj_feasible=info.get("obj_feasible"),
                      cpp_cons=info.get("cons"), cpp_pool_gap=info.get("pool_gap"),
                      f_ref_tau0=f_ref0, solver="cpp"))
        rows.append(m); flush()
        print(f"  r={r:4d} red={reduction:5.1f}% dF={m['delta_F_pct']:8.3f} "
              f"GapF(vs tau0)={m['Gap_F_pct']:+9.5f} R2={m['link_R2']:.4f} cpu={secs:.1f}s",
              flush=True)

    # decision-A reference: best feasible over {tau0 full, all rank solves}
    objs = [f_ref0] + [float(x["cpp_obj_feasible"]) for x in rows
                       if x.get("cpp_obj_feasible") not in (None, "")]
    f_best = min(objs)
    for x in rows:
        o = x.get("cpp_obj_feasible")
        x["Gap_F_final_pct"] = 100.0 * (float(o) - f_best) / f_best if o not in (None, "") else ""
        x["f_ref_best"] = f_best
    flush()
    print(f"\n[T5-rank] f_best={f_best:,.4f}")
    for x in rows:
        print(f"  r={x['rank']:4d} GapF_final={float(x['Gap_F_final_pct']):+.5f}% "
              f"R2={x['link_R2']:.4f} cpu={x['cpu_s']:.1f}s")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
