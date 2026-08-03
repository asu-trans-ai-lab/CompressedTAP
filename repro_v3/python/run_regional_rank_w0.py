"""ONE C++ experiment on Chicago Regional: rank x offset (author scope, 2026-07-25).

Chicago Regional, E0 baseline pool (n=2,024,525 paths, ~30 links per path), tau=1.03,
C++ solver (compressed_solver_o3sse), single thread. Two factors:

    rank r in {10, 20, 50}
    offset  w0 present  vs  w0 removed exactly (v_base -= B2'w0, d_eff += A2 w0, w0 = 0)

Falsifiable prediction from the cost rule 2r <~ links-per-path: with ~30 links per path,
r = 10-15 should win and r = 50 should lose. The w0 arm re-tests the load-bearing claim on a
large real network in the compiled engine (so far shown on the grid and on Sioux Falls).

Baseline is the tau=0 full C++ solve on the same instance, same session.

    python run_regional_rank_w0.py [--ranks 10,20,50] [--tau 1.03]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CERTPY = ROOT / "vendor"
sys.path.insert(0, str(HERE))
from run_grid_w0free import make_w0free, solve as cpp_solve

ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
DS = ROOT / "data" / "chicago_regional"
POOL = "path_pool_E0_baseline.csv"
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "regional_rank_w0"


def export(tau, rank, out_dir):
    if (out_dir / "meta.json").exists():
        return json.loads((out_dir / "meta.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    r = subprocess.run([sys.executable, str(CERTPY / "export_stage7.py"), str(DS), POOL,
                        str(out_dir), str(rank), "1", str(tau)],
                       capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"export failed: {r.stderr[-400:]}")
    meta = json.loads(r.stdout.strip().splitlines()[-1])
    meta["export_wall_s"] = round(time.perf_counter() - t, 1)
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="10,20,50")
    ap.add_argument("--tau", type=float, default=1.03)
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]
    out = HERE.parent / "results" / "regional_rank_w0_cpp.csv"
    rows = []

    def flush():
        keys = sorted({k for r_ in rows for k in r_})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r_ in rows:
                w.writerow({k: r_.get(k, "") for k in keys})

    print("[regional] exporting tau=0 (full baseline)", flush=True)
    m0 = export(0.0, 0, SCRATCH / "tau0")
    n = int(m0["n"])
    nnz_per_path = int(m0["nnz"]) / n
    print(f"  n={n:,} od={m0['n_od']:,} nnz/path={nnz_per_path:.1f} "
          f"-> rule says 2r <~ {nnz_per_path:.0f}, i.e. r <~ {nnz_per_path/2:.0f}", flush=True)

    tfull, inf_f = cpp_solve(SCRATCH / "tau0", "full", 0)
    ofull = float(inf_f["obj_feasible"])
    print(f"  FULL t={tfull:.1f}s obj={ofull:,.2f} inner={inf_f.get('inner_iters')}",
          flush=True)
    rows.append(dict(arm="full", r="", t_s=round(tfull, 1), speedup="", obj=ofull,
                     gap_pct="", inner=inf_f.get("inner_iters"), svd_s=""))
    flush()

    objs = [ofull]
    for rk in ranks:
        d = SCRATCH / f"tau{a.tau}_r{rk}"
        meta = export(a.tau, rk, d)
        s, r_used = int(meta["n_major"]), int(meta["r"])
        red = 100 * (n - s - r_used) / n
        svd = meta.get("svd_time_s", 0)
        print(f"[regional] r={rk}: majors={s:,} red={red:.1f}% svd={svd:.1f}s", flush=True)

        t, inf = cpp_solve(d, "hard", rk)
        o = float(inf["obj_feasible"]); objs.append(o)
        print(f"  w0      t={t:8.1f}s SPEEDUP={tfull/t:5.2f}x obj={o:,.2f} "
              f"inner={inf.get('inner_iters')}", flush=True)
        rows.append(dict(arm="w0", r=rk, t_s=round(t, 1), speedup=round(tfull/t, 2),
                         obj=o, gap_pct="", inner=inf.get("inner_iters"),
                         reduction_pct=round(red, 1), svd_s=svd))
        flush()

        dfree = SCRATCH / f"tau{a.tau}_r{rk}_w0free"
        make_w0free(d, dfree)
        t2, inf2 = cpp_solve(dfree, "hard", rk)
        o2 = float(inf2["obj_feasible"]); objs.append(o2)
        print(f"  w0-free t={t2:8.1f}s SPEEDUP={tfull/t2:5.2f}x obj={o2:,.2f} "
              f"inner={inf2.get('inner_iters')}", flush=True)
        rows.append(dict(arm="w0free", r=rk, t_s=round(t2, 1), speedup=round(tfull/t2, 2),
                         obj=o2, gap_pct="", inner=inf2.get("inner_iters"),
                         reduction_pct=round(red, 1), svd_s=svd))
        flush()

    best = min(objs)
    for r_ in rows:
        if r_.get("obj"):
            r_["gap_pct"] = round(100 * (float(r_["obj"]) - best) / best, 4)
    flush()

    print(f"\n[regional] SUMMARY (best feasible = {best:,.2f}; full baseline {tfull:.1f}s)")
    print(f"  {'r':>4} {'w0 speedup':>11} {'w0 gap%':>9} {'w0free speedup':>15} {'w0free gap%':>12}")
    for rk in ranks:
        a1 = next((x for x in rows if x["arm"] == "w0" and x["r"] == rk), None)
        a2 = next((x for x in rows if x["arm"] == "w0free" and x["r"] == rk), None)
        print(f"  {rk:>4} {a1['speedup']:>11.2f} {a1['gap_pct']:>9.4f} "
              f"{a2['speedup']:>15.2f} {a2['gap_pct']:>12.4f}")
    print(f"wrote {out}\n[regional] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
