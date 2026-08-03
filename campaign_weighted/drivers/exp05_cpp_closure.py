"""Experiment 5 -- C++ closure demonstration on the two real road networks.

The C++ record on real networks is weak: best Chicago Sketch speedup 1.50x (E0, r=10), and
Chicago Regional never solved compressed at all. But the two laws established in this study
both point at one untested corner:

  * richness: at fixed r=50 the C++ Sketch speedup rises with K/OD (0.34, 0.78, 0.81, 0.75,
    0.99 for Kbar = 2.45 .. 15.34) -- it approaches parity but never clears it;
  * rank: at fixed richness, dropping r nearly doubles it (E0: 0.80x at r=50 -> 1.50x at r=10),
    and the cost rule says 2r <~ links-per-path.

High K/OD combined with SMALL r has never been run in C++. That is this experiment.

  Chicago Sketch, richest pools (K10, K15), r in {6, 10, 20}
  Chicago Regional, E0 pool (2.02M paths, 30 links/path -> rule says r <~ 15), r in {8, 15}

Fresh uncompressed baseline per instance in the same session, so every ratio is
within-session and within-engine. Weighted basis (the campaign default).

    python drivers/exp05_cpp_closure.py [--sketch K10,K15] [--sketch-ranks 6,10,20]
                                        [--regional-ranks 8,15]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "cpp_closure"
SKETCH_DIR = C.DATA / "chicago_sketch"
POOLS = {"E2": "path_pool.csv", "K10": "path_pool_K10.csv", "K15": "path_pool_K15.csv"}
REGIONAL_DIR = C.DATA / "chicago_regional"
REGIONAL_POOL = "path_pool_E0_baseline.csv"
TAU_SKETCH, TAU_REGIONAL = 4.54, 1.03


def run_instance(label, dsdir, pool, tau, ranks, rows):
    K.banner(f"exp05 {label}: {pool}  tau={tau}  ranks={ranks}")
    d0 = SCRATCH / label / "tau0"
    m0 = K.export(dsdir, pool, d0, 0, 0.0, False)
    n, nnz = int(m0["n"]), int(m0["nnz"])
    print(f"  n={n:,} od={m0['n_od']:,} nnz/path={nnz/n:.1f} "
          f"-> rule: r <~ {nnz/n/2:.0f}", flush=True)
    tf, of, inf_f = K.solve_cpp(d0, "full", 0)
    print(f"  FULL      t={tf:8.1f}s obj={of:,.2f} inner={inf_f.get('inner_iters')}",
          flush=True)
    rows.append(dict(instance=label, n=n, nnz_per_path=round(nnz / n, 1), r="",
                     t_s=round(tf, 1), speedup="", obj=of, gap_pct=""))
    K.write("exp05_cpp_closure", rows)
    objs = [of]
    for r in ranks:
        dc = SCRATCH / label / f"r{r}"
        meta = K.export(dsdir, pool, dc, r, tau, True)      # weighted basis
        s = int(meta["n_major"])
        red = 100.0 * (n - s - int(meta["r"])) / n
        tc, oc, inf = K.solve_cpp(dc, "hard", r)
        objs.append(oc)
        best = min(objs)
        print(f"  r={r:<3d}      t={tc:8.1f}s SPEEDUP={tf/tc:5.2f}x "
              f"gap={100*(oc-best)/best:+8.4f}% red={red:.1f}% "
              f"inner={inf.get('inner_iters')}", flush=True)
        rows.append(dict(instance=label, n=n, nnz_per_path=round(nnz / n, 1), r=r,
                         t_s=round(tc, 1), speedup=round(tf / tc, 2), obj=oc,
                         reduction_pct=round(red, 1), inner=inf.get("inner_iters"),
                         svd_s=meta.get("svd_time_s")))
        K.write("exp05_cpp_closure", rows)
    # restate gaps against the best objective seen on this instance
    best = min(objs)
    for x in rows:
        if x["instance"] == label and x.get("obj"):
            x["gap_pct"] = round(100 * (float(x["obj"]) - best) / best, 4)
    K.write("exp05_cpp_closure", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", default="K10,K15")
    ap.add_argument("--sketch-ranks", default="6,10,20")
    ap.add_argument("--regional-ranks", default="8,15")
    a = ap.parse_args()
    rows = []
    for tag in [t.strip() for t in a.sketch.split(",") if t.strip()]:
        run_instance(f"sketch-{tag}", SKETCH_DIR, POOLS[tag], TAU_SKETCH,
                     [int(x) for x in a.sketch_ranks.split(",")], rows)
    if a.regional_ranks.strip():
        run_instance("regional-E0", REGIONAL_DIR, REGIONAL_POOL, TAU_REGIONAL,
                     [int(x) for x in a.regional_ranks.split(",")], rows)

    print("\n[exp05] SUMMARY -- best C++ speedup per real network")
    for inst in sorted({x["instance"] for x in rows}):
        cand = [x for x in rows if x["instance"] == inst and x["r"] != ""]
        if not cand:
            continue
        b = max(cand, key=lambda x: x["speedup"])
        print(f"  {inst:14s} best {b['speedup']:5.2f}x at r={b['r']:<3} "
              f"gap={b['gap_pct']:+.4f}%  (n={b['n']:,}, {b['nnz_per_path']} links/path)")
    print(f"wrote {C.RESULTS/'exp05_cpp_closure.csv'}\n[exp05] DONE", flush=True)


if __name__ == "__main__":
    main()
