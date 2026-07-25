"""Table 4 Panel B (tab:cutoff-summary, panel B) — route-richness scaling on Chicago Sketch.

Fixed network (Chicago Sketch) and FIXED OD set; K in {8,16,32,64} paths/OD, r=50. For each K
the same OD set is solved full and compressed by BOTH ALM and gradient projection (GP-signed,
v3's own representation), and the compression speedups + feasible gap are reported.

Design (all choices flagged for author review):
  * Candidate paths come from the project's deterministic penalty-KSP generator (the same one
    behind SFK10/25/60), run on the Sketch network. Per OD: free-flow shortest path, then rounds
    of multiplying the last path's link costs by `penalty` and re-solving.
  * FIXED OD set = the multi-path ODs for which the generator yields at least max(K)=64 distinct
    paths, so every K column uses the identical OD set (the manuscript's "fixed OD set").
  * For each K, each OD keeps its K lowest-free-flow-cost paths (deterministic truncation).
  * KSP extras carry no nominal flow, which would collapse the affine box w0+U_r z>=0 to {0}
    (the C2 phenomenon). As in sf_rich_pools.py, x0 is set by a logit split of demand over the
    pool by base cost, giving every minor path a small positive nominal flow.
  * q represented/OD = (s+r)/ell; K/q = n/(s+r) is the compression factor. Majors s come from a
    fixed flow threshold so q stays ~constant as K grows (the scaling the panel isolates).

    python run_table4B_richness.py [--n-od 800] [--penalty 1.4] [--rank 50] [--tau 1e-3]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CERTPY = ROOT / "source" / "updated_TAPLite" / "python"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CERTPY))

import v3_metrics as M
import compressed_assignment as ca
from run_table2_consistency import op_gp_full, LMOIndex
from gp_compressed import solve_gp_signed


def make_cert(P):
    """Relative FW duality gap on a full path-flow x -- the SAME certificate op_gp_full stops
    on, so full and compressed GP are compared at matched accuracy."""
    lmo = LMOIndex(P["p2od"], P["n_od"], P["n"])
    v0 = P.get("v0", 0.0)
    B, bpr, d = P["B"], P["bpr"], P["d"]

    def cert(x_full):
        v = v0 + np.asarray(B.T @ x_full).flatten()
        c = np.asarray(B @ bpr.t(v)).flatten()
        s_lmo = lmo(c, d)
        return float(c @ (x_full - s_lmo)) / max(abs(float(c @ x_full)), 1e-12)

    return cert

SKETCH = CERTPY.parent / "data" / "03_chicago_sketch"
KS = [8, 16, 32, 64]


def ksp_paths(n_od_cap, penalty, kmax):
    """Deterministic penalty-KSP on the Sketch network. Returns {(o,d): [pathpaths]} where
    each path is a tuple of link_ids, ordered by free-flow cost, for ODs with >= kmax paths."""
    link = pd.read_csv(SKETCH / "link.csv", low_memory=False)
    node = pd.read_csv(SKETCH / "node.csv")
    n2i = {n: i for i, n in enumerate(node.node_id.to_numpy())}
    nn = len(node)
    fr = link.from_node_id.map(n2i).to_numpy()
    to = link.to_node_id.map(n2i).to_numpy()
    if "vdf_fftt" in link.columns:
        fftt = link.vdf_fftt.to_numpy(float)
    else:
        fftt = (60 * link.length / link.free_speed).to_numpy(float)
    lid = link.link_id.to_numpy()
    edge_of = {}
    for e, (a, b) in enumerate(zip(fr, to)):
        edge_of.setdefault((a, b), e)
    dem = pd.read_csv(SKETCH / "demand.csv")
    dem = dem[(dem.volume > 0) & (dem.o_zone_id != dem.d_zone_id)]
    # highest-demand ODs first, so the fixed set is the busiest ODs
    dem = dem.sort_values("volume", ascending=False)
    ods = dem[["o_zone_id", "d_zone_id"]].drop_duplicates().to_numpy()

    def extract(pred, o_i, d_i):
        path, cur, g = [], d_i, 0
        while cur != o_i and g < 4000:
            p = pred[cur]
            if p < 0:
                return None
            e = edge_of.get((p, cur))
            if e is None:
                return None
            path.append(e); cur = p; g += 1
        return path[::-1]

    out = {}
    base_G = csr_matrix((fftt, (fr, to)), shape=(nn, nn))
    pred_cache = {}
    scanned = 0
    for o, d in ods:
        if len(out) >= n_od_cap:
            break
        scanned += 1
        o_i, d_i = n2i[o], n2i[d]
        if o_i not in pred_cache:
            _, pred_cache[o_i] = dijkstra(base_G, indices=o_i, return_predecessors=True)
        p1 = extract(pred_cache[o_i], o_i, d_i)
        if p1 is None:
            continue
        found = {tuple(p1): float(sum(fftt[e] for e in p1))}
        cost = fftt.copy(); last = p1
        for _ in range(kmax * 3):  # extra rounds; many collapse to duplicates
            if len(found) >= kmax:
                break
            for e in last:
                cost[e] *= penalty
            _, pk = dijkstra(csr_matrix((cost, (fr, to)), shape=(nn, nn)),
                             indices=o_i, return_predecessors=True)
            pth = extract(pk, o_i, d_i)
            if pth is None:
                break
            last = pth
            t = tuple(pth)
            if t not in found:
                found[t] = float(sum(fftt[e] for e in pth))
        if len(found) >= kmax:
            paths = sorted(found, key=lambda t: found[t])[:kmax]
            out[(int(o), int(d))] = [tuple(int(lid[e]) for e in p) for p in paths]
    print(f"  penalty-KSP: scanned {scanned} ODs, kept {len(out)} with >= {kmax} paths",
          flush=True)
    return out


def write_pool(kpaths, K, path):
    rows = []
    for (o, d), paths in kpaths.items():
        for p in paths[:K]:
            rows.append({"o_zone_id": o, "d_zone_id": d,
                         "link_ids": ";".join(str(x) for x in p),
                         "prob_ref": np.nan, "volume_ref": np.nan, "source": "ksp"})
    pd.DataFrame(rows).to_csv(path, index=False)


def set_logit_x0(P):
    cost = np.asarray(P["B"] @ P["bpr"].t0).flatten()
    cmin = np.full(P["n_od"], np.inf); np.minimum.at(cmin, P["p2od"], cost)
    w = np.exp(-(cost - cmin[P["p2od"]]) / np.maximum(0.15 * cmin[P["p2od"]], 1e-6))
    ws = np.zeros(P["n_od"]); np.add.at(ws, P["p2od"], w)
    P["x0"] = P["d"][P["p2od"]] * w / ws[P["p2od"]]


def timed(fn):
    t = time.perf_counter(); r = fn(); return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-od", type=int, default=800)
    ap.add_argument("--penalty", type=float, default=1.4)
    ap.add_argument("--rank", type=int, default=50)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--no-gp", action="store_true",
                    help="skip the (slow, non-scaling) compressed GP; ALM speedup diagnostic")
    a = ap.parse_args()

    print(f"[T4B] penalty-KSP on Sketch (n_od cap {a.n_od}, kmax {max(KS)})", flush=True)
    kpaths = ksp_paths(a.n_od, a.penalty, max(KS))
    if len(kpaths) < 20:
        print(f"  only {len(kpaths)} ODs reached K=64 -- Sketch may be too small for K=64 "
              f"on this OD set; consider a larger --n-od or smaller Kmax.", flush=True)

    scratch = Path(__import__("os").environ.get("TMP", "/tmp")) / "v3_panelB"
    scratch.mkdir(parents=True, exist_ok=True)
    out = HERE.parent / "results" / "table4B_richness.csv"
    rows = []

    for K in KS:
        pool_csv = scratch / f"sketch_K{K}.csv"
        write_pool(kpaths, K, pool_csv)
        P = ca.load_problem(str(SKETCH), str(pool_csv), multi_path_only=True)
        set_logit_x0(P)
        sl = M.od_slice_index(P)
        ell = P["n_od"]; n = P["n"]
        major = ca.split_major_minor(P, tau=a.tau)
        s = int(major.sum())
        C = ca.build_compressed(P, major, a.rank)
        sr = s + C["r"]
        q = sr / ell
        print(f"[T4B] K={K}: n={n:,} ell={ell} s={s} r={C['r']} s+r={sr} "
              f"q={q:.2f} K/q={n/sr:.2f}", flush=True)

        # ALM full vs compressed
        (solA_f, tA_f) = timed(lambda: ca.solve_full(P, tol=a.tol, max_outer=40))
        (solA_c, tA_c) = timed(lambda: ca.solve_compressed(P, C, regime="hard",
                                                           tol=a.tol, max_outer=40))
        # GP full vs compressed (signed), matched on the same relative-gap certificate
        if a.no_gp:
            tG_f = tG_c = float("nan")
        else:
            cert = make_cert(P)
            (gpf, tG_f) = timed(lambda: op_gp_full(P, a.tol, max_seconds=600))
            (gpc, tG_c) = timed(lambda: solve_gp_signed(P, C, cert, tol=a.tol,
                                                        max_seconds=600))

        # feasible gap: compressed ALM vs full ALM reference (best feasible of the two)
        xf = M.convert_euclid(solA_f["x_raw"], P["d"], P["p2od"])
        vf = M.link_flow(P, xf); f_full = float(P["bpr"].beckmann(vf))
        xc = M.convert_euclid(solA_c["x_raw"], P["d"], P["p2od"])
        vc = M.link_flow(P, xc); f_comp = float(P["bpr"].beckmann(vc))
        f_ref = min(f_full, f_comp)
        gap = 100.0 * (f_comp - f_ref) / f_ref

        rows.append(dict(K=K, q=round(q, 2), K_over_q=round(n / sr, 2), n=n, s_plus_r=sr,
                         majors=s, rank=C["r"], ell=ell,
                         alm_full_s=round(tA_f, 2), alm_comp_s=round(tA_c, 2),
                         alm_speedup=round(tA_f / max(tA_c, 1e-9), 2),
                         gp_full_s=round(tG_f, 2), gp_comp_s=round(tG_c, 2),
                         gp_speedup=round(tG_f / max(tG_c, 1e-9), 2),
                         feasible_gap_pct=round(gap, 4)))
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"  ALM {tA_f:.1f}/{tA_c:.1f}s x{rows[-1]['alm_speedup']}  "
              f"GP {tG_f:.1f}/{tG_c:.1f}s x{rows[-1]['gp_speedup']}  "
              f"gap {gap:+.4f}%", flush=True)

    print(f"\n[T4B] DONE -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
