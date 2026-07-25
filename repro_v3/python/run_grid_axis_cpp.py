"""Grid mechanism case (author request 2026-07-25): effect of compression as the grid grows
and as column-generation (K-shortest-path) depth grows -- ALL C++.

  network : N x N Manhattan grid, bidirectional links, t0=1 min, cap=600, BPR(0.15, 4)
  demand  : left column -> right column, all N^2 pairs, q = 720/N per OD
            (mid-cut v/c held ~1.2 at every N, so congestion is constant across sizes)
  pools   : C++ ksp_gen.exe with K_extra in the axis (penalty-KSP, deterministic)
  x0      : logit split of q over the pool by cost_base (theta = 0.15 c_min) -- the same
            fix as Panel B, without it the KSP extras carry zero flow and the affine box
            collapses
  split   : tau = 75th percentile of x0 (fixed rule -> ~25% majors), r = 20
  solves  : compressed_solver_o3sse.exe full vs hard, single thread, within-engine speedup

    python run_grid_axis_cpp.py [--Ns 8,16,24,32] [--Ks 4,8,16,32,64]
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
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CERTPY = ROOT / "source" / "updated_TAPLite" / "python"
CPPDIR = ROOT / "stable_release" / "10_implementation"
KSP = CPPDIR / "ksp_gen.exe"
SOLVER = CPPDIR / "compressed_solver_o3sse.exe"
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "grid_axis"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
RANK = 20


def gen_grid(N, out_dir):
    """Emit GMNS node/link/demand for an N x N grid, left col -> right col demand."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nid = lambda i, j: i * N + j + 1                     # 1-based node ids
    nodes = [{"node_id": nid(i, j), "x_coord": j, "y_coord": i, "zone_id": nid(i, j)}
             for i in range(N) for j in range(N)]
    pd.DataFrame(nodes).to_csv(out_dir / "node.csv", index=False)
    links, lid = [], 0
    for i in range(N):
        for j in range(N):
            for di, dj in ((0, 1), (1, 0)):
                ii, jj = i + di, j + dj
                if ii < N and jj < N:
                    for a, b in ((nid(i, j), nid(ii, jj)), (nid(ii, jj), nid(i, j))):
                        lid += 1
                        links.append({"link_id": lid, "from_node_id": a, "to_node_id": b,
                                      "length": 1.0, "free_speed": 60.0, "lanes": 1,
                                      "capacity": 600.0, "vdf_fftt": 1.0,
                                      "vdf_alpha": 0.15, "vdf_beta": 4.0})
    pd.DataFrame(links).to_csv(out_dir / "link.csv", index=False)
    q = 720.0 / N
    dem = [{"o_zone_id": nid(i, 0), "d_zone_id": nid(k, N - 1), "volume": q}
           for i in range(N) for k in range(N)]
    pd.DataFrame(dem).to_csv(out_dir / "demand.csv", index=False)
    return len(links), len(dem), q


def logit_x0(pool_csv, q):
    """volume_ref = q * softmax(-cost/theta) per OD, theta = 0.15 * c_min."""
    df = pd.read_csv(pool_csv)
    out = []
    for (_, _), g in df.groupby(["o_zone_id", "d_zone_id"], sort=False):
        c = g["cost_base"].to_numpy(float)
        w = np.exp(-(c - c.min()) / max(0.15 * c.min(), 1e-6))
        out.append(q * w / w.sum())
    df["volume_ref"] = np.concatenate(out)
    df["prob_ref"] = df["volume_ref"] / q
    df.to_csv(pool_csv, index=False)
    return df


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
        raise RuntimeError(f"no RESULT {mode}: {r.stdout[-300:]}")
    return secs, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="8,16,24,32")
    ap.add_argument("--Ks", default="4,8,16,32,64")
    a = ap.parse_args()
    out = HERE.parent / "results" / "grid_axis_cpp.csv"
    rows = []

    def flush():
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    for N in [int(x) for x in a.Ns.split(",")]:
        gdir = SCRATCH / f"g{N}"
        m, n_od, q = gen_grid(N, gdir)
        print(f"\n[grid] N={N}: {m} links, {n_od} ODs, q={q:.1f}", flush=True)
        for K in [int(x) for x in a.Ks.split(",")]:
            pool = gdir / f"pool_K{K}.csv"
            # always regenerate: a stale/partial pool (e.g. from an interrupted run)
            # silently poisons kbar/tau/x0
            r = subprocess.run([str(KSP), str(gdir), str(K), "1.4", str(pool)],
                               capture_output=True, text=True, env=ENV)
            if r.returncode != 0:
                print(f"  K={K}: ksp_gen failed: {r.stderr[-200:]}"); continue
            df = logit_x0(pool, q)
            assert df["volume_ref"].notna().all(), "x0 has NaNs"
            kbar = len(df) / n_od
            tau = float(np.quantile(df["volume_ref"].to_numpy(float), 0.75))
            exp0 = gdir / f"K{K}_tau0"; expc = gdir / f"K{K}_tauC"
            for d, rk, tv in ((exp0, 0, 0.0), (expc, RANK, tau)):
                if not (d / "meta.json").exists():
                    rr = subprocess.run(
                        [sys.executable, str(CERTPY / "export_stage7.py"), str(gdir),
                         pool.name, str(d), str(rk), "1", str(tv)],
                        capture_output=True, text=True, env=ENV)
                    if rr.returncode != 0:
                        raise RuntimeError(f"export failed: {rr.stderr[-300:]}")
            meta = json.loads((expc / "meta.json").read_text())
            n, s, r_used = int(meta["n"]), int(meta["n_major"]), int(meta["r"])
            red = 100 * (n - s - r_used) / n
            tf, inf_f = solve(exp0, "full", 0)
            th, inf_h = solve(expc, "hard", RANK)
            of, oh = float(inf_f["obj_feasible"]), float(inf_h["obj_feasible"])
            best = min(of, oh)
            su = tf / th
            print(f"  K={K:3d}: n={n:6,} Kbar={kbar:5.2f} red={red:5.1f}% "
                  f"full={tf:6.1f}s hard={th:6.1f}s SPEEDUP={su:5.2f}x "
                  f"gap={100*(oh-best)/best:+.4f}%", flush=True)
            rows.append(dict(N=N, K_extra=K, n=n, n_od=n_od, kbar=round(kbar, 2),
                             majors=s, rank=r_used, reduction_pct=round(red, 1),
                             t_full_s=round(tf, 2), t_hard_s=round(th, 2),
                             speedup=round(su, 2), obj_full=of, obj_hard=oh,
                             gap_hard_pct=round(100*(oh-best)/best, 4),
                             inner_full=inf_f.get("inner_iters"),
                             inner_hard=inf_h.get("inner_iters"), tau=tau))
            flush()

    print("\n[grid] SPEEDUP(N, K) matrix:")
    Ns = sorted({r_["N"] for r_ in rows}); Ks = sorted({r_["K_extra"] for r_ in rows})
    print("        " + "".join(f"K={k:<7}" for k in Ks))
    for N in Ns:
        line = f"  N={N:<4}"
        for k in Ks:
            m_ = [r_ for r_ in rows if r_["N"] == N and r_["K_extra"] == k]
            line += f"{m_[0]['speedup']:<9.2f}" if m_ else " " * 9
        print(line)
    print(f"wrote {out}\n[grid] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
