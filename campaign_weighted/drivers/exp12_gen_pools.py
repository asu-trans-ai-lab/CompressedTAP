"""Experiment 12, stage 1 -- generate the path-rich, OD-light extreme case.

Chicago Sketch topology, a FIXED set of the top-M OD pairs by demand volume, and NESTED
penalty-KSP pools P_8 c P_16 c P_32 c P_64:

  * one deterministic ksp_gen run at depth 64 (shortest path, then 63 rounds of x1.4
    penalty on the last path's links);
  * per-OD DEDUPLICATION by link sequence (penalty rounds can regenerate a path),
    keeping first-generation order;
  * path-length filter cost_base <= RATIO x the OD's shortest cost_base (default 1.5);
  * nested truncation to the first 8/16/32/64 surviving paths per OD -- nesting holds
    by construction because the order is fixed before truncation;
  * per-OD logit nominal flow on each pool (each OD's own demand; theta as exp03),
    because KSP extras carry no flow and the affine box would collapse without it.

The demand file contains ONLY the selected ODs. This is a controlled scaling instance --
the OD count and OD-level work are fixed while the full-space variable count grows
linearly in K -- not a model of the real Sketch demand. Disclosed wherever reported.

    python drivers/exp12_gen_pools.py [--M 20000] [--K 64] [--ratio 1.5]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "extreme_case"
SRC = C.DATA / "03_chicago_sketch"
DEPTHS = [8, 16, 32, 64]


def logit_per_od(df, dvol, theta_frac=0.15):
    """volume_ref = q_od * logit(cost_base) within each OD (exp03 convention,
    per-OD demand instead of a uniform q)."""
    out = []
    for (o, d), g in df.groupby(["o_zone_id", "d_zone_id"], sort=False):
        c = g["cost_base"].to_numpy(float)
        w = np.exp(-(c - c.min()) / max(theta_frac * c.min(), 1e-6))
        out.append(dvol[(o, d)] * w / w.sum())
    df = df.copy()
    df["volume_ref"] = np.concatenate(out)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=20000)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--ratio", type=float, default=1.5)
    a = ap.parse_args()
    ds = SCRATCH / f"M{a.M}"
    ds.mkdir(parents=True, exist_ok=True)
    K.banner(f"exp12 pool generation: top {a.M:,} ODs, nested depths {DEPTHS}, "
             f"ratio<={a.ratio}")

    # ---- dataset: full topology, top-M demand only
    for f in ("node.csv", "link.csv"):
        (ds / f).write_bytes((SRC / f).read_bytes())
    dem = pd.read_csv(SRC / "demand.csv")
    dem = dem.groupby(["o_zone_id", "d_zone_id"], as_index=False).volume.sum()
    dem = dem.sort_values("volume", ascending=False).head(a.M).reset_index(drop=True)
    dem.to_csv(ds / "demand.csv", index=False)
    dvol = {(r.o_zone_id, r.d_zone_id): r.volume for r in dem.itertuples()}
    print(f"  demand: {len(dem):,} ODs, total volume {dem.volume.sum():,.0f} "
          f"(share of full demand: "
          f"{dem.volume.sum()/pd.read_csv(SRC/'demand.csv').volume.sum():.1%})",
          flush=True)

    # ---- one deterministic deep KSP run
    raw = ds / f"pool_raw_K{a.K}.csv"
    print(f"  running ksp_gen depth {a.K} (deterministic, penalty 1.4) ...", flush=True)
    r = subprocess.run([str(C.KSP), str(ds), str(a.K), "1.4", str(raw)],
                       capture_output=True, text=True, env=C.env())
    assert r.returncode == 0, f"ksp_gen failed: {r.stderr[-400:]}"
    df = pd.read_csv(raw)
    print(f"  raw pool: {len(df):,} paths", flush=True)

    # ---- dedupe (keep first occurrence) + length-ratio filter, order preserved
    df["__key"] = df.link_ids.astype(str)
    before = len(df)
    df = df.drop_duplicates(subset=["o_zone_id", "d_zone_id", "__key"], keep="first")
    print(f"  dedupe: {before:,} -> {len(df):,}", flush=True)
    cmin = df.groupby(["o_zone_id", "d_zone_id"])["cost_base"].transform("min")
    before = len(df)
    df = df[df.cost_base <= a.ratio * cmin].reset_index(drop=True)
    print(f"  ratio filter (<= {a.ratio} x shortest): {before:,} -> {len(df):,}",
          flush=True)
    df["__rank"] = df.groupby(["o_zone_id", "d_zone_id"]).cumcount()

    # ---- nested truncations + per-OD logit nominal flow
    for depth in DEPTHS:
        sub = df[df["__rank"] < depth].drop(columns=["__key", "__rank"]).copy()
        sub = logit_per_od(sub, dvol)
        sub["prob_ref"] = sub.volume_ref / sub.groupby(
            ["o_zone_id", "d_zone_id"]).volume_ref.transform("sum").clip(lower=1e-300) \
            * 1.0
        out = ds / f"pool_X{depth}.csv"
        sub.to_csv(out, index=False)
        kbar = len(sub) / sub.groupby(["o_zone_id", "d_zone_id"]).ngroups
        print(f"  pool_X{depth}: {len(sub):,} paths, Kbar={kbar:.2f}, "
              f"ceiling 1-1/Kbar = {100*(1-1/kbar):.1f}%", flush=True)

    print(f"\n  dataset dir: {ds}\n[exp12-gen] DONE", flush=True)


if __name__ == "__main__":
    main()
