"""Experiment 3 -- the grid mechanism family, and the offset ablation on it. ALL C++.

N x N Manhattan grid, left column -> right column demand (q = 720/N keeps the mid-cut v/c
near 1.2 at every N), penalty-KSP pools of depth K, logit nominal flow, threshold at the
GRID_TAU_Q quantile, rank GRID_RANK. For each (N, K) cell and each basis arm: full versus
compressed, then the same compressed instance with the affine offset removed exactly.

Output: results/exp03_grid.csv
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "campaign_weighted_grid"


def gen_grid(N, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    nid = lambda i, j: i * N + j + 1
    pd.DataFrame([{"node_id": nid(i, j), "x_coord": j, "y_coord": i, "zone_id": nid(i, j)}
                  for i in range(N) for j in range(N)]).to_csv(out_dir / "node.csv",
                                                              index=False)
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
    q = C.GRID_Q_PER_OD / N
    pd.DataFrame([{"o_zone_id": nid(i, 0), "d_zone_id": nid(k, N - 1), "volume": q}
                  for i in range(N) for k in range(N)]).to_csv(out_dir / "demand.csv",
                                                              index=False)
    return len(links), N * N, q


def logit_x0(pool_csv, q):
    """KSP extras carry no nominal flow; without this the affine box collapses to {0}."""
    df = pd.read_csv(pool_csv)
    out = []
    for _, g in df.groupby(["o_zone_id", "d_zone_id"], sort=False):
        c = g["cost_base"].to_numpy(float)
        w = np.exp(-(c - c.min()) / max(0.15 * c.min(), 1e-6))
        out.append(q * w / w.sum())
    df["volume_ref"] = np.concatenate(out)
    df["prob_ref"] = df["volume_ref"] / q
    df.to_csv(pool_csv, index=False)
    return df


def main():
    rows = []
    for N in C.GRID_NS:
        gdir = SCRATCH / f"g{N}"
        m, n_od, q = gen_grid(N, gdir)
        K.banner(f"exp03 grid N={N}: {m} links, {n_od} ODs, q={q:.1f}")
        for Kd in C.GRID_KS:
            pool = gdir / f"pool_K{Kd}.csv"
            # always regenerate: a stale pool silently poisons the threshold and x0
            r = subprocess.run([str(C.KSP), str(gdir), str(Kd), "1.4", str(pool)],
                               capture_output=True, text=True, env=C.env())
            if r.returncode != 0:
                print(f"  K={Kd}: ksp_gen failed"); continue
            df = logit_x0(pool, q)
            tau = float(np.quantile(df["volume_ref"].to_numpy(float), C.GRID_TAU_Q))
            kbar = len(df) / n_od

            d0 = gdir / f"K{Kd}_tau0"
            K.export(gdir, pool.name, d0, 0, 0.0, False)
            tf, of, _ = K.solve_cpp(d0, "full", 0)

            for basis in C.BASES:
                dc = gdir / f"K{Kd}_{basis}"
                meta = K.export(gdir, pool.name, dc, C.GRID_RANK, tau,
                                basis == "weighted")
                n, s = int(meta["n"]), int(meta["n_major"])
                red = 100 * (n - s - int(meta["r"])) / n
                tc, oc, inf = K.solve_cpp(dc, "hard", C.GRID_RANK)
                dfree = gdir / f"K{Kd}_{basis}_nooffset"
                K.make_offset_free(dc, dfree)
                tz, oz, _ = K.solve_cpp(dfree, "hard", C.GRID_RANK)
                best = min(of, oc, oz)
                print(f"  K={Kd:3d} {basis:11s} full={tf:6.1f}s comp={tc:6.1f}s "
                      f"speedup={tf/tc:5.2f}x gap={100*(oc-best)/best:7.4f}%  | "
                      f"no-offset {tf/tz:5.2f}x gap={100*(oz-best)/best:7.4f}%", flush=True)
                rows.append(dict(N=N, K=Kd, n=n, kbar=round(kbar, 2), basis=basis,
                                 reduction_pct=round(red, 1), t_full_s=round(tf, 1),
                                 t_comp_s=round(tc, 1), speedup=round(tf / tc, 2),
                                 gap_pct=round(100 * (oc - best) / best, 4),
                                 t_nooffset_s=round(tz, 1),
                                 speedup_nooffset=round(tf / tz, 2),
                                 gap_nooffset_pct=round(100 * (oz - best) / best, 4),
                                 tau=round(tau, 4)))
                K.write("exp03_grid", rows)
    print(f"\nwrote {C.RESULTS/'exp03_grid.csv'}\n[exp03] DONE", flush=True)


if __name__ == "__main__":
    main()
