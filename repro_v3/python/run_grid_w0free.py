"""w0 ablation on the grid matrix (author request 2026-07-25): remove the affine offset
(w0 = 0) from the minor representation and rerun -- ALL C++.

The offset-free variant keeps the SAME split, U, D, M and moves the offset flow out of the
base terms so the model stays exactly consistent:

    x_minor = U z          (was  w0 + U z)
    v_base' = v_base - B2^T w0        (minor nominal flow no longer pre-loaded)
    d_eff'  = d_eff  + A2  w0        (its demand must now be carried by y and z)
    w0'     = 0

Implemented as a blob-level post-fixer on the cached export dirs (B CSR + major mask +
p2od are all in the dump), so the certified export path is untouched. Each grid cell's
hard solve is rerun on the fixed export and compared to the stored w0 result.

    python run_grid_w0free.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOLVER = ROOT / "stable_release" / "10_implementation" / "compressed_solver_o3sse.exe"
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "grid_axis"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def make_w0free(src: Path, dst: Path):
    if (dst / "meta.json").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy(f, dst / f.name)
    meta = json.loads((dst / "meta.json").read_text())
    n = int(meta["n"])
    indptr = np.fromfile(dst / "B_indptr.i64", dtype=np.int64)
    indices = np.fromfile(dst / "B_indices.i32", dtype=np.int32)
    major = np.fromfile(dst / "major.u8", dtype=np.uint8).astype(bool)
    p2od = np.fromfile(dst / "p2od.i32", dtype=np.int32)
    x0m = np.fromfile(dst / "x0m.f64", dtype=np.float64)
    v_base = np.fromfile(dst / "v_base.f64", dtype=np.float64)
    d_eff = np.fromfile(dst / "d_eff.f64", dtype=np.float64)
    minors = np.where(~major)[0]
    assert len(minors) == len(x0m), (len(minors), len(x0m))
    # v_base -= B2^T w0  (scatter each minor's nominal flow off its links)
    for k, p in enumerate(minors):
        if x0m[k] != 0.0:
            v_base[indices[indptr[p]:indptr[p + 1]]] -= x0m[k]
    # d_eff += A2 w0  (its demand goes back to the explicit variables)
    np.add.at(d_eff, p2od[minors], x0m)
    v_base.tofile(dst / "v_base.f64")
    d_eff.tofile(dst / "d_eff.f64")
    np.zeros_like(x0m).tofile(dst / "x0m.f64")


def solve(dump, mode, rank):
    t = time.perf_counter()
    r = subprocess.run([str(SOLVER), str(dump), mode, "40", str(rank), "", "300"],
                       capture_output=True, text=True, env=ENV)
    secs = time.perf_counter() - t
    info = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("RESULT"):
            for tok in ln.split():
                if "=" in tok:
                    k, v = tok.split("=", 1); info[k] = v
    if "obj_feasible" not in info:
        raise RuntimeError(f"no RESULT: {r.stdout[-300:]}")
    return secs, info


def main():
    base = {(int(r["N"]), int(r["K_extra"])): r for r in csv.DictReader(
        (HERE.parent / "results" / "grid_axis_cpp.csv").open())}
    out = HERE.parent / "results" / "grid_w0free_cpp.csv"
    rows = []
    for (N, K), b in sorted(base.items()):
        src = SCRATCH / f"g{N}" / f"K{K}_tauC"
        if not (src / "meta.json").exists():
            print(f"  N={N} K={K}: export missing, skip"); continue
        dst = SCRATCH / f"g{N}" / f"K{K}_w0free"
        make_w0free(src, dst)
        secs, info = solve(dst, "hard", int(b["rank"]))
        o_free = float(info["obj_feasible"])
        o_full = float(b["obj_full"]); o_w0 = float(b["obj_hard"])
        best = min(o_full, o_w0, o_free)
        su = float(b["t_full_s"]) / secs
        rows.append(dict(N=N, K_extra=K,
                         t_w0free_s=round(secs, 2), speedup_w0free=round(su, 2),
                         gap_w0free_pct=round(100 * (o_free - best) / best, 4),
                         speedup_w0=b["speedup"], gap_w0_pct=b["gap_hard_pct"],
                         obj_w0free=o_free, inner_w0free=info.get("inner_iters")))
        print(f"  N={N:2d} K={K:3d}: w0free {su:5.2f}x gap={rows[-1]['gap_w0free_pct']:+9.4f}%"
              f"   (w0: {b['speedup']}x gap={b['gap_hard_pct']}%)", flush=True)
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}\n[w0free] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
