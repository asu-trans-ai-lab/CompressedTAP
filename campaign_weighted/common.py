"""Shared machinery for the flow-weighted campaign: loading, solving, metrics, CSV output.

Every experiment goes through these functions, so a metric is computed one way only.
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config as C

sys.path.insert(0, str(C.CERTPY))
sys.path.insert(0, str(C.ROOT / "git_dev" / "CompressedTAP" / "repro_v3" / "python"))

import compressed_assignment as ca          # noqa: E402
import v3_metrics as M                      # noqa: E402


# ------------------------------------------------------------------ instances
def load(name):
    d, pool, tau, _ = C.ROAD[name]
    P = ca.load_problem(str(d), pool, multi_path_only=True)
    return P, tau


def compress(P, tau, rank, basis):
    """basis in {'weighted','unweighted'}; weighted uses the nominal flow as the weight."""
    major = ca.split_major_minor(P, tau=tau)
    w = P["x0"] if basis == "weighted" else None
    Cc = ca.build_compressed(P, major, rank, weight_flows=w)
    s = int(major.sum())
    Cc["reduction_pct"] = 100.0 * (P["n"] - s - Cc["r"]) / P["n"]
    Cc["majors"] = s
    return Cc


# ------------------------------------------------------------------ solving
def timed(fn, reps=1):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), out, ts


def feas_obj(P, x_raw):
    xf = M.convert_euclid(np.asarray(x_raw, float), P["d"], P["p2od"])
    return float(P["bpr"].beckmann(M.link_flow(P, xf)))


def solve_full_py(P, reps=1):
    t, sol, _ = timed(lambda: ca.solve_full(P, tol=C.TOL, max_outer=C.MAX_OUTER_FULL), reps)
    return t, feas_obj(P, sol["x_raw"]), sol


def solve_comp_py(P, Cc, reps=1, regime="hard"):
    t, sol, _ = timed(lambda: ca.solve_compressed(P, Cc, regime=regime, tol=C.TOL,
                                                  max_outer=C.MAX_OUTER_COMP), reps)
    return t, feas_obj(P, sol["x_raw"]), sol


# ------------------------------------------------------------------ C++ path
def export(dsdir, pool, out_dir, rank, tau, weighted):
    """export_stage7 argv: <dir> <pool> <out> <rank> <multi> <tau> [weighted]"""
    if (out_dir / "meta.json").exists():
        return json.loads((out_dir / "meta.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(C.CERTPY / "export_stage7.py"), str(dsdir), str(pool),
           str(out_dir), str(rank), "1", str(tau)]
    if weighted:
        cmd.append("weighted")
    r = subprocess.run(cmd, capture_output=True, text=True, env=C.env())
    if r.returncode != 0:
        raise RuntimeError("export failed: " + r.stderr[-400:])
    meta = json.loads(r.stdout.strip().splitlines()[-1])
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def solve_cpp(dump_dir, mode, rank, reps=1):
    def once():
        t = time.perf_counter()
        r = subprocess.run([str(C.CPP), str(dump_dir), mode, str(C.MAX_OUTER_FULL),
                            str(rank), "", str(C.MAX_INNER)],
                           capture_output=True, text=True, env=C.env())
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
    ts, info = [], None
    for _ in range(reps):
        s, info = once(); ts.append(s)
    return statistics.median(ts), float(info["obj_feasible"]), info


def make_offset_free(src: Path, dst: Path):
    """Exact removal of the affine offset: v_base -= B2'w0, d_eff += A2 w0, w0 = 0."""
    import shutil
    if (dst / "meta.json").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy(f, dst / f.name)
    indptr = np.fromfile(dst / "B_indptr.i64", dtype=np.int64)
    indices = np.fromfile(dst / "B_indices.i32", dtype=np.int32)
    major = np.fromfile(dst / "major.u8", dtype=np.uint8).astype(bool)
    p2od = np.fromfile(dst / "p2od.i32", dtype=np.int32)
    x0m = np.fromfile(dst / "x0m.f64", dtype=np.float64)
    v_base = np.fromfile(dst / "v_base.f64", dtype=np.float64)
    d_eff = np.fromfile(dst / "d_eff.f64", dtype=np.float64)
    minors = np.where(~major)[0]
    for k, p in enumerate(minors):
        if x0m[k] != 0.0:
            v_base[indices[indptr[p]:indptr[p + 1]]] -= x0m[k]
    np.add.at(d_eff, p2od[minors], x0m)
    v_base.tofile(dst / "v_base.f64"); d_eff.tofile(dst / "d_eff.f64")
    np.zeros_like(x0m).tofile(dst / "x0m.f64")


# ------------------------------------------------------------------ output
def write(name, rows):
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    p = C.RESULTS / f"{name}.csv"
    keys = sorted({k for r in rows for k in r})
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return p


def banner(msg):
    print(f"\n{'='*72}\n{msg}\n{'='*72}", flush=True)
