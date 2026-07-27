"""Experiment 13 -- the Regional extreme case, ALL C++ (the ceiling law's prediction test).

Regional-E0 (Kbar=7.48) is a certified structural negative: the reduction ceiling
1 - 1/Kbar = 86.6% sits below the ~90% paying regime. The law PREDICTS that the same
network with a rich pool must cross break-even. exp12's generator built that pool set
(top-20,000 ODs, 26.7% of Regional demand -- a controlled path-richness instance on the
real topology, NOT full-demand Regional; nested X8 c X16 c X32 c X64, ratio 2.5).

Both arms run the compiled solver (single thread), so every ratio is within-engine:

  FULL        compressed_solver full mode (tau=0 export)
  COMPRESSED  compressed_solver hard mode, weighted basis, r=6, tau = 75th percentile
              of volume_ref  -- the INCUMBENT global-SVD representation; the dynamic
              atom hybrid exists only in Python and is NOT used here (stated wherever
              these results appear; a C++ atom port is the named follow-up if the hard
              mode cannot reach the tighter target)

Metric: the solver's own HIST_CSV trace (elapsed_s, pool_gap), the SAME certificate in
both modes, on the feasibility-projected iterate. Pre-registered moderate targets
G* in {0.020, 0.015} -- above the certified representation floor (~0.013), so both are
in principle reachable by the incumbent representation. Time-to-target is read from the
RUNNING MINIMUM of the trace, rescaled to measured wall time; a mode that never reaches
a target is reported DNS, never credited.

    python drivers/exp13_regional_cpp.py [--depths 8,16,32,64] [--rank 6] [--M 20000]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "extreme_case"
TARGETS = (0.020, 0.015)


def run_cpp(dump, mode, rank, hist, max_outer, max_inner):
    env = C.env()
    env["HIST_CSV"] = str(hist)
    t = time.perf_counter()
    p = subprocess.run([str(C.CPP), str(dump), mode, str(max_outer), str(rank),
                        "", str(max_inner)],
                       capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t
    info = {}
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT"):
            for tok in ln.split():
                if "=" in tok:
                    kk, v = tok.split("=", 1)
                    info[kk] = v
    if "obj_feasible" not in info:
        raise RuntimeError(f"no RESULT ({mode}): {p.stdout[-400:]}")
    H = np.zeros((0, 2))
    if Path(hist).exists():
        try:
            H = np.loadtxt(hist, delimiter=",", skiprows=1, ndmin=2)
            H = H[np.isfinite(H).all(axis=1)]
        except Exception:
            pass
    return wall, info, H


def t_at(H, wall, final_gap, target):
    """First wall time the running-min gap trace reaches <= target; None if never."""
    if H.size:
        g = np.minimum.accumulate(H[:, 1])
        k = np.where(g <= target)[0]
        if k.size:
            span = H[-1, 0]
            return round(float(H[k[0], 0]) * (wall / span if span > 0 else 1.0), 1)
    return round(wall, 1) if final_gap <= target else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="regional")
    ap.add_argument("--M", type=int, default=20000)
    ap.add_argument("--depths", default="8,16,32,64")
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--tau-q", type=float, default=0.75)
    ap.add_argument("--max-outer", type=int, default=40)
    ap.add_argument("--max-inner", type=int, default=300)
    a = ap.parse_args()
    ds = SCRATCH / a.net / f"M{a.M}"
    rows = []
    tag = f"{a.net}_M{a.M}_cpp"
    K.banner(f"exp13 [{a.net}] extreme ladder, ALL C++: full vs hard(r={a.rank}), "
             f"targets {TARGETS}\n  instance: top-{a.M:,} ODs, controlled richness "
             f"(NOT full-demand {a.net})")

    for depth in [int(x) for x in a.depths.split(",")]:
        pool = f"pool_X{depth}.csv"
        assert (ds / pool).exists(), f"missing {pool}: run exp12_gen_pools first"
        df = pd.read_csv(ds / pool, usecols=["volume_ref"])
        tau = float(np.quantile(df.volume_ref.to_numpy(float), a.tau_q))
        scr = ds / f"cpp_X{depth}"

        d0 = scr / "tau0"
        m0 = K.export(ds, pool, d0, 0, 0.0, False)
        n, ell = int(m0["n"]), int(m0["n_od"])
        K.banner(f"X{depth}: n={n:,} od={ell:,} Kbar={n/ell:.2f} tau={tau:.4f}")
        tf, inf_f, Hf = run_cpp(d0, "full", 0, scr / "hist_full.csv",
                                a.max_outer, a.max_inner)
        gf = float(inf_f.get("pool_gap", "nan"))
        print(f"  FULL  t={tf:8.1f}s  obj={float(inf_f['obj_feasible']):,.2f}  "
              f"final_gap={gf:.4e}", flush=True)

        dc = scr / f"r{a.rank}"
        mc = K.export(ds, pool, dc, a.rank, tau, True)          # weighted basis + w0
        red = 100.0 * (n - int(mc["n_major"]) - int(mc["r"])) / n
        tc, inf_c, Hc = run_cpp(dc, "hard", a.rank, scr / "hist_hard.csv",
                                a.max_outer, a.max_inner)
        gc = float(inf_c.get("pool_gap", "nan"))
        print(f"  HARD  t={tc:8.1f}s  obj={float(inf_c['obj_feasible']):,.2f}  "
              f"final_gap={gc:.4e}  red={red:.1f}%  "
              f"svd={mc.get('svd_time_s')}s", flush=True)

        line = dict(depth=depth, n=n, n_od=ell, kbar=round(n / ell, 2), tau=round(tau, 4),
                    reduction_pct=round(red, 1), t_full_wall_s=round(tf, 1),
                    t_hard_wall_s=round(tc, 1), gap_full_final=gf, gap_hard_final=gc,
                    obj_full=float(inf_f["obj_feasible"]),
                    obj_hard=float(inf_c["obj_feasible"]),
                    svd_s=mc.get("svd_time_s"))
        for t in TARGETS:
            Tf = t_at(Hf, tf, gf, t)
            Tc = t_at(Hc, tc, gc, t)
            line[f"T_full@{t}"] = Tf
            line[f"T_comp@{t}"] = Tc
            line[f"S@{t}"] = round(Tf / Tc, 2) if Tf and Tc else None
            print(f"  X{depth}  G*={t}:  T_full={Tf}  T_comp={Tc}  S={line[f'S@{t}']}",
                  flush=True)
        rows.append(line)
        K.write(f"exp13_{tag}", rows)

    print("\n[exp13] LADDER SUMMARY")
    for r in rows:
        print(f"  X{r['depth']:<3d} Kbar={r['kbar']:6.2f} red={r['reduction_pct']:5.1f}%  "
              f"S(.020)={r['S@0.02']}  S(.015)={r['S@0.015']}")
    print(f"wrote {C.RESULTS / f'exp13_{tag}.csv'}\n[exp13] DONE", flush=True)


if __name__ == "__main__":
    main()
