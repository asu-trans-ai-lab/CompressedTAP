"""C++ engine on the central K/OD axis (engine-independence check).

Same instances as run_kod_axis.py (Chicago Sketch pools V2/E0/E2/K10/K15, tau=4.54, r=50),
solved by the tuned C++ solver (compressed_solver_o3sse.exe: -O3 baseline-SSE2, verified
bit-identical objectives to the certified -O2 build). Per pool: export tau=0 (full) and
tau=4.54 r=50 (hard) via the certified export_stage7, then time both modes, single thread.

    python run_kod_axis_cpp.py [--pools V2,E0,E2,K10,K15]
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
ROOT = HERE.parents[3]
CERTPY = ROOT / "source" / "updated_TAPLite" / "python"
sys.path.insert(0, str(HERE))
from run_kod_axis import POOLS, TAU

CPP = ROOT / "stable_release" / "10_implementation" / "compressed_solver_o3sse.exe"
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "v3_cpp_export" / "kaxis"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def export(dsdir, pool, tau, rank, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "meta.json").exists():
        return json.loads((out_dir / "meta.json").read_text())
    r = subprocess.run([sys.executable, str(CERTPY / "export_stage7.py"), str(dsdir), pool,
                        str(out_dir), str(rank), "1", str(tau)],
                       capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"export failed: {r.stderr[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def solve(dump_dir, mode, rank):
    t = time.perf_counter()
    r = subprocess.run([str(CPP), str(dump_dir), mode, "40", str(rank), "", "300"],
                       capture_output=True, text=True, env=ENV)
    secs = time.perf_counter() - t
    info = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("RESULT"):
            for tok in ln.split():
                if "=" in tok:
                    k, v = tok.split("=", 1); info[k] = v
    if "obj_feasible" not in info:
        raise RuntimeError(f"no RESULT ({mode}): {r.stdout[-300:]}")
    return secs, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="V2,E0,E2,K10,K15")
    a = ap.parse_args()
    out = HERE.parent / "results" / "kod_axis_cpp.csv"
    rows = []

    def flush():
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    for tag in [p.strip() for p in a.pools.split(",")]:
        dsdir, pool = POOLS[tag]
        print(f"\n[cpp-axis] {tag}: exporting", flush=True)
        m0 = export(dsdir, pool, 0.0, 0, SCRATCH / tag / "tau0")
        mc = export(dsdir, pool, TAU, 50, SCRATCH / tag / "tauC")
        n = int(mc["n"]); s = int(mc["n_major"]); r_used = int(mc["r"])
        kbar = n / int(mc["n_od"])
        red = 100 * (n - s - r_used) / n
        print(f"  n={n:,} K-bar={kbar:.2f} red={red:.1f}%", flush=True)

        tf, inf_f = solve(SCRATCH / tag / "tau0", "full", 0)
        print(f"  FULL t={tf:7.1f}s obj={float(inf_f['obj_feasible']):,.2f} "
              f"inner={inf_f.get('inner_iters')}", flush=True)
        th, inf_h = solve(SCRATCH / tag / "tauC", "hard", 50)
        of, oh = float(inf_f["obj_feasible"]), float(inf_h["obj_feasible"])
        best = min(of, oh)
        su = tf / th
        print(f"  HARD t={th:7.1f}s SPEEDUP={su:5.2f}x gap={100*(oh-best)/best:+.4f}% "
              f"inner={inf_h.get('inner_iters')}", flush=True)
        rows.append(dict(pool=tag, n=n, kbar=round(kbar, 2), reduction_pct=round(red, 1),
                         t_full_s=round(tf, 1), t_hard_s=round(th, 1),
                         speedup=round(su, 2), obj_full=of, obj_hard=oh,
                         gap_hard_vs_best_pct=round(100*(oh-best)/best, 4),
                         inner_full=inf_f.get("inner_iters"),
                         inner_hard=inf_h.get("inner_iters")))
        flush()

    print("\n[cpp-axis] C++ CENTRAL CURVE:")
    for r_ in rows:
        print(f"  {r_['pool']:4s} K-bar={r_['kbar']:6.2f} red={r_['reduction_pct']:5.1f}% "
              f"SPEEDUP={r_['speedup']:5.2f}x gap={r_['gap_hard_vs_best_pct']:+.4f}%")
    print(f"wrote {out}\n[cpp-axis] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
