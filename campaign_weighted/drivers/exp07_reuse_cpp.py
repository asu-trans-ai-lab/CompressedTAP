"""Experiment 7 -- C++ basis reuse on a real network, against MATCHED-ACCURACY baselines.

Three things this fixes relative to exp04 (Python, grid) and exp05 (C++, no reuse):

  1. ENGINE. Everything here is the compiled solver, weighted basis, w0 from the pool split
     -- the exp05 configuration that produced 3.06x / 3.26x.

  2. REUSE IS LITERAL. The compressed dump is built ONCE. Each scenario copies that directory
     and patches only the perturbed quantity (dvec.f64 for demand, bpr_cap.f64 for capacity,
     and d_eff.f64 = d_eff + (d_new - d_old) since w0 is unchanged). U.f64, D.f64, M.f64,
     x0m.f64 and v_base.f64 are byte-identical across every scenario, which is checked.

  3. PERFECT BASELINES. exp05's uncompressed runs stopped at an iteration cap and finished
     WORSE than the compressed runs (K10: full 16,972,896 vs compressed 16,926,325), so every
     ratio there is against an under-converged reference. Here each solve exports HIST_CSV,
     a (elapsed_s, pool_gap) trace, and speedups are read off at a COMMON pool_gap target --
     the tightest gap every method on that scenario actually reached. No method is credited
     for stopping early, and the target is reported with the ratio.

The compressed solve gets NO warm start (its x0.f64 is the baseline pool split); the
uncompressed comparator DOES (x0.f64 <- the baseline solution). The bias is against us.

    python drivers/exp07_reuse_cpp.py [--pool K10] [--rank 6] [--scenarios d10,c10]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "reuse_cpp"
SKETCH = C.DATA / "chicago_sketch"
POOLS = {"K10": "path_pool_K10.csv", "K15": "path_pool_K15.csv"}
TAU = 4.54
# files that MUST be identical across scenarios if the basis is genuinely reused
FROZEN = ["U.f64", "D.f64", "M.f64", "x0m.f64", "v_base.f64", "major.u8"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def run(dump, mode, rank, hist, xraw=None, max_outer=None):
    """One solve. Returns (wall_s, RESULT dict, hist array of (t, pool_gap))."""
    env = C.env()
    env["HIST_CSV"] = str(hist)
    if xraw:
        env["VDUMP_XRAW"] = str(xraw)
    cmd = [str(C.CPP), str(dump), mode, str(max_outer or C.MAX_OUTER_FULL),
           str(rank), "", str(C.MAX_INNER)]
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    secs = time.perf_counter() - t
    info = {}
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT"):
            for tok in ln.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    info[k] = v
    if "obj_feasible" not in info:
        raise RuntimeError(f"no RESULT ({mode}) in {dump}: {p.stdout[-400:]}")
    H = np.zeros((0, 2))
    if Path(hist).exists():
        try:
            H = np.loadtxt(hist, delimiter=",", skiprows=1, ndmin=2)
        except Exception:
            pass
    return secs, info, H


def t_at(H, target, wall, final_gap):
    """Wall time at which the trace first reaches pool_gap <= target.

    The trace is sampled every 20 inner iterations, so it is scaled to the measured wall
    time (the trace clock excludes setup). If the target is never reached in the trace but
    the FINAL reported gap meets it, the full wall time is returned; otherwise None.
    """
    if H.size:
        ok = np.where(H[:, 1] <= target)[0]
        if ok.size:
            span = H[-1, 0]
            return float(H[ok[0], 0]) * (wall / span if span > 0 else 1.0)
    return wall if final_gap <= target else None


def perturb(base_dir, out_dir, kind, delta, seed):
    """Copy a dump and patch ONLY the perturbed quantity. Returns out_dir.

    The SAME seed is used for the uncompressed and compressed copies of a scenario, so both
    solvers see the identical perturbed instance.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir)
    rng = np.random.default_rng(seed)
    if kind == "demand":
        d_old = np.fromfile(out_dir / "dvec.f64", dtype=np.float64)
        d_new = d_old * (1.0 + rng.uniform(-delta, delta, d_old.size))
        d_new.tofile(out_dir / "dvec.f64")
        f = out_dir / "d_eff.f64"
        if f.exists():                       # compressed dump: shift by the SAME delta
            (np.fromfile(f, dtype=np.float64) + (d_new - d_old)).tofile(f)
    else:
        cap = np.fromfile(out_dir / "bpr_cap.f64", dtype=np.float64)
        (cap * (1.0 + rng.uniform(-delta, delta, cap.size))).tofile(out_dir / "bpr_cap.f64")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="K10")
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--scenarios", default="d05,d10,c05,c10")
    ap.add_argument("--max-outer", type=int, default=60)
    a = ap.parse_args()
    tag = a.pool
    root = SCRATCH / tag
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    K.banner(f"exp07 basis reuse, C++ weighted, Chicago Sketch {tag}, r={a.rank}, "
             f"tau={TAU}\n  matched-accuracy baselines via HIST_CSV pool_gap traces")

    # ---------------------------------------------------------- baseline scenario
    d_full = root / "base_full"
    m_full = K.export(SKETCH, POOLS[tag], d_full, 0, 0.0, False)
    n, n_od, m_links = int(m_full["n"]), int(m_full["n_od"]), int(m_full["m"])
    t0 = time.perf_counter()
    d_comp = root / "base_comp"
    m_comp = K.export(SKETCH, POOLS[tag], d_comp, a.rank, TAU, True)   # weighted + w0
    t_prep = float(m_comp.get("svd_time_s") or (time.perf_counter() - t0))
    red = 100.0 * (n - int(m_comp["n_major"]) - int(m_comp["r"])) / n
    print(f"  n={n:,} od={n_od:,} links={m_links:,} | majors={int(m_comp['n_major']):,} "
          f"r={m_comp['r']} reduction={red:.1f}% | T_prep(SVD)={t_prep:.2f}s", flush=True)
    frozen_ref = {f: sha(d_comp / f) for f in FROZEN}

    x_base = root / "x_base.f64"
    tb, ib, Hb = run(d_full, "full", 0, root / "hist_base.csv", xraw=x_base,
                     max_outer=a.max_outer)
    print(f"  BASELINE full  t={tb:7.1f}s obj={float(ib['obj_feasible']):,.2f} "
          f"pool_gap={float(ib['pool_gap']):.3e}", flush=True)
    rows.append(dict(pool=tag, scenario="baseline", method="full-cold", t_s=round(tb, 1),
                     obj=float(ib["obj_feasible"]), pool_gap=float(ib["pool_gap"]),
                     t_prep_s=round(t_prep, 2), reduction_pct=round(red, 1)))
    K.write("exp07_reuse_cpp", rows)
    d_ref = np.fromfile(d_full / "dvec.f64", dtype=np.float64)

    # ---------------------------------------------------------- follow-up scenarios
    for i, sc in enumerate([s.strip() for s in a.scenarios.split(",") if s.strip()]):
        kind = "demand" if sc[0] == "d" else "capacity"
        delta = int(sc[1:]) / 100.0
        seed = 100 + i
        perturb(d_full, root / f"{sc}_full", kind, delta, seed)
        sc_dir = perturb(d_comp, root / f"{sc}_comp", kind, delta, seed)
        # reuse is literal: the basis blobs must be untouched
        bad = [f for f in FROZEN if sha(sc_dir / f) != frozen_ref[f]]
        assert not bad, f"basis NOT reused, changed: {bad}"

        # warm-started uncompressed: same dump, x0 <- baseline solution
        wdir = root / f"{sc}_warm"
        if wdir.exists():
            shutil.rmtree(wdir)
        shutil.copytree(root / f"{sc}_full", wdir)
        shutil.copy(x_base, wdir / "x0.f64")

        res = {}
        for meth, dd, mode, rk in (("full-cold", root / f"{sc}_full", "full", 0),
                                   ("full-warm", wdir, "full", 0),
                                   ("compressed-reuse", sc_dir, "hard", a.rank)):
            t, info, H = run(dd, mode, rk, root / f"hist_{sc}_{meth}.csv",
                             max_outer=a.max_outer)
            res[meth] = (t, float(info["obj_feasible"]),
                         float(info.get("pool_gap", "nan")), H)
            print(f"  {sc:5s} {meth:17s} t={t:7.1f}s obj={float(info['obj_feasible']):,.2f} "
                  f"pool_gap={float(info.get('pool_gap','nan')):.3e}", flush=True)

        # matched accuracy: the tightest pool_gap ALL three actually reached
        gaps = [v[2] for v in res.values() if np.isfinite(v[2])]
        target = max(gaps) if gaps else float("nan")
        tm = {k: t_at(v[3], target, v[0], v[2]) for k, v in res.items()}
        best = min(v[1] for v in res.values())
        for meth, (t, o, pg, _) in res.items():
            rows.append(dict(
                pool=tag, scenario=sc, kind=kind, delta=delta, method=meth,
                t_s=round(t, 1), obj=o, pool_gap=pg,
                gap_vs_best_pct=round(100 * (o - best) / best, 4),
                target_pool_gap=target,
                t_at_target_s=None if tm[meth] is None else round(tm[meth], 1)))
        tw, tc = tm.get("full-warm"), tm.get("compressed-reuse")
        if tw and tc:
            print(f"  {sc:5s} -> at matched pool_gap {target:.3e}: warm {tw:.1f}s vs "
                  f"compressed {tc:.1f}s = {tw/tc:.2f}x", flush=True)
            rows[-1]["speedup_vs_warm_matched"] = round(tw / tc, 2)
            rows[-1]["speedup_vs_cold_matched"] = (
                round(tm["full-cold"] / tc, 2) if tm.get("full-cold") else None)
        K.write("exp07_reuse_cpp", rows)

    # ---------------------------------------------------------- break-even
    fu = [r for r in rows if r.get("scenario") != "baseline" and r.get("t_at_target_s")]
    tw = [r["t_at_target_s"] for r in fu if r["method"] == "full-warm"]
    tc = [r["t_at_target_s"] for r in fu if r["method"] == "compressed-reuse"]
    if tw and tc:
        mw, mc = float(np.mean(tw)), float(np.mean(tc))
        print(f"\n  T_prep={t_prep:.2f}s | warm-full {mw:.1f}s vs compressed {mc:.1f}s "
              f"(matched accuracy)")
        if mw > mc:
            H = int(np.ceil(t_prep / (mw - mc)))
            print(f"  saving {mw-mc:.1f}s/solve -> BREAK-EVEN H* = {H}")
        else:
            H = None
            print(f"  compressed NOT faster ({mw-mc:+.1f}s): no break-even")
        rows.append(dict(pool=tag, scenario="SUMMARY", t_prep_s=round(t_prep, 2),
                         t_warm_mean_s=round(mw, 1), t_compressed_mean_s=round(mc, 1),
                         speedup_vs_warm_matched=round(mw / mc, 2),
                         break_even_H=H if H else "none"))
        K.write("exp07_reuse_cpp", rows)
    print(f"\nwrote {C.RESULTS/'exp07_reuse_cpp.csv'}\n[exp07] DONE", flush=True)


if __name__ == "__main__":
    main()
