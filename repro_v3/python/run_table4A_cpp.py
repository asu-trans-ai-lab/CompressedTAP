"""Table 4 Panel A via the certified C++ solver — the feasible path for large networks.

The Python driver (run_table4A_thresholds.py) cannot finish the 4.8M-path networks: each
ALM outer iteration is 20-40 min (R16). The C++ compressed_solver is the SAME SPG-ALM
signed-SVD algorithm (certify_cpp_r1) and ~10-25x faster. This driver:

  for each (network, tau):
    1. export the instance+threshold to the C++ binary format via export_stage7.py
       (which loads through the certified load_problem, so it is byte-consistent with the
       P used here for metrics);
    2. run compressed_solver <dir> hard <max_outer> <rank> '' <max_inner> with
       VDUMP_XRAW set, so C++ dumps the UNPROJECTED reconstructed path flows;
    3. read x_raw back and compute delta_F / Gap_F / R2 through the ONE shared v3_metrics,
       exactly as the Python driver does -- so no reported number changes meaning.

Reference (decision a): the tau=0 C++ full solve IS v^ref. Gate 2 (Gap_F >= 0) then holds
by construction for the tau=0 row. For tau>0 rows, Gap_F is measured against that same
reference; if any comes out negative it means the reference is the looser solve and
recompute_gapf.py resolves it from the objectives on disk.

    python run_table4A_cpp.py --net regional [--rank 50] [--max-outer 40] [--max-inner 300]
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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CERTPY = ROOT / "source" / "updated_TAPLite" / "python"
CPPDIR = ROOT / "stable_release" / "10_implementation"
CPPEXE = CPPDIR / "compressed_solver.exe"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CERTPY))

import v3_metrics as M
import compressed_assignment as ca

DATA = ROOT / "source" / "updated_TAPLite" / "data"
NETS = {
    "sketch": dict(dir=DATA / "03_chicago_sketch", pool="path_pool.csv", multi=1,
                   taus=[0.0, 0.46, 1.06, 4.54]),
    "regional": dict(dir=DATA / "04_chicago_regional", pool="path_pool.csv", multi=1,
                     taus=[0.0, 0.23, 0.46, 1.03]),
    "philadelphia": dict(dir=ROOT / "OR_paper_revision_V2" / "m4_rerun" / "data"
                         / "philadelphia", pool="pool.csv", multi=1,
                         taus=[0.0, 0.67, 0.99, 5.33]),
}
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "v3_cpp_export"


def export(dsdir, pool, multi, tau, rank, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(CERTPY / "export_stage7.py"), str(dsdir), pool,
           str(out_dir), str(rank), str(multi), str(tau)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
    if r.returncode != 0:
        raise RuntimeError("export_stage7 failed: " + r.stderr[-500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def cpp_solve(dump_dir, mode, max_outer, rank, max_inner):
    xraw = dump_dir / "x_raw.f64"
    if xraw.exists():
        xraw.unlink()
    t = time.perf_counter()
    r = subprocess.run([str(CPPEXE), str(dump_dir), mode, str(max_outer), str(rank),
                        "", str(max_inner)], capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                            "VDUMP_XRAW": str(xraw)})
    secs = time.perf_counter() - t
    if not xraw.exists():
        raise RuntimeError("C++ produced no x_raw. stdout tail:\n" + r.stdout[-800:])
    x = np.fromfile(xraw, dtype=np.float64)
    # parse the RESULT line for inner_iters / obj / cons
    info = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("RESULT"):
            for tok in ln.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    info[k] = v
    return x, secs, info, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=list(NETS))
    ap.add_argument("--rank", type=int, default=50)
    ap.add_argument("--max-outer", type=int, default=40)
    ap.add_argument("--max-inner", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    cfg = NETS[a.net]
    if not CPPEXE.exists():
        print(f"C++ solver not built at {CPPEXE}"); return 1

    print(f"[T4A-cpp] loading {a.net} for metrics", flush=True)
    P = ca.load_problem(str(cfg["dir"]), cfg["pool"], multi_path_only=bool(cfg["multi"]))
    sl = M.od_slice_index(P)
    print(f"  n={P['n']:,} od={P['n_od']:,} links={P['B'].shape[1]:,}", flush=True)

    out = HERE.parent / "results" / f"table4A_{a.net}_cpp.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush(rows):
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # ---- reference (decision a): the tau=0 C++ full solve
    exp0 = SCRATCH / a.net / "tau0"
    print("[T4A-cpp] reference = C++ tau=0 full solve", flush=True)
    meta0 = export(cfg["dir"], cfg["pool"], cfg["multi"], 0.0, 0, exp0)  # rank 0: full mode, no SVD
    x0raw, t0, info0, _ = cpp_solve(exp0, "full", a.max_outer, 0, a.max_inner)
    x0f = M.convert_euclid(x0raw, P["d"], P["p2od"])
    v_ref = M.link_flow(P, x0f); f_ref = float(P["bpr"].beckmann(v_ref))
    print(f"  f_ref={f_ref:,.4f} t={t0:.1f}s inner={info0.get('inner_iters')}", flush=True)

    rows = []
    n = P["n"]
    for tau in cfg["taus"]:
        print(f"[T4A-cpp] tau={tau}", flush=True)
        if tau <= 0.0:
            x_raw, secs, info = x0raw, t0, info0
            s_major, r_used, reduction = n, 0, 0.0
        else:
            expd = SCRATCH / a.net / f"tau{tau}"
            meta = export(cfg["dir"], cfg["pool"], cfg["multi"], tau, a.rank, expd)
            s_major = int(meta["n_major"]); r_used = int(meta["r"])
            reduction = 100.0 * (n - s_major - r_used) / n
            x_raw, secs, info, _ = cpp_solve(expd, "hard", a.max_outer, a.rank, a.max_inner)
        m = M.evaluate_both(P, x_raw, f_ref, v_ref, sl)
        m.update(dict(net=a.net, tau=tau, n_paths=n, n_od=P["n_od"], majors=s_major,
                      rank=r_used, reduction_pct=reduction,
                      raw_obj_diff_pct=M.raw_objective_diff_pct(P, x_raw, f_ref),
                      inner_iters=info.get("inner_iters"), cpu_s=secs,
                      cpp_obj_feasible=info.get("obj_feasible"),
                      cpp_cons=info.get("cons"), cpp_pool_gap=info.get("pool_gap"),
                      solver="cpp", ref_source="cpp-tau0-full"))
        rows.append(m); flush(rows)
        print(f"  red={reduction:5.1f}%  dF={m['delta_F_pct']:7.4f}%  "
              f"GapF={m['Gap_F_pct']:+9.5f}%  R2={m['link_R2']:.4f}  cpu={secs:.1f}s",
              flush=True)

    neg = [r for r in rows if r["Gap_F_pct"] < -1e-6]
    print(f"\nwrote {out}", flush=True)
    print(f"GATE 2: {'PASS' if not neg else 'FAIL (recompute_gapf resolves)'}"
          + ("" if not neg else f" {[(r['tau'], round(r['Gap_F_pct'],5)) for r in neg]}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
