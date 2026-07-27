"""Experiment 6 -- can Chicago Regional be made to pay by compressing harder?

exp05 found Regional-E0 at tau=1.03 does NOT pay in C++ (<0.58x). The diagnosis was that
the instance was never at the favourable corner: 84% reduction, 1.7M minor rows against
325k majors, whereas the Sketch cells that reached 3.06-3.26x had 90-92% reduction.

That is a statement about the THRESHOLD, not the rank, so this sweeps tau upward at fixed
small rank and reports the reduction each tau actually delivers before solving. If Regional
still fails at 90%+ reduction, the negative is about the network and not about the tuning,
and that is the finding.

Baseline: the tau=0 full C++ solve measured in exp05 on this machine, 1085.3s (objective
18,854,545.32, reproduced bit for bit across three separate runs of 785s, 988s and 1085s).
Reusing it rather than re-measuring saves ~18 min; the ratio is flagged as cross-run.

    python drivers/exp06_regional_tau.py [--taus 3,8,20] [--rank 8] [--cap-s 900]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as K
import config as C

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "regional_tau"
DS = C.DATA / "04_chicago_regional"
POOL = "path_pool_E0_baseline.csv"
T_FULL = 1085.3          # exp05, same machine; deterministic objective 18,854,545.32
O_FULL = 18854545.32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="3,8,20")
    ap.add_argument("--rank", type=int, default=8)
    a = ap.parse_args()
    rows = []
    K.banner(f"exp06 Chicago Regional (E0 pool) -- tau sweep at r={a.rank}\n"
             f"  baseline: {T_FULL}s full C++ solve (exp05, cross-run)")

    for tau in [float(x) for x in a.taus.split(",")]:
        d = SCRATCH / f"tau{tau}_r{a.rank}"
        meta = K.export(DS, POOL, d, a.rank, tau, True)      # weighted basis
        n, s, r_used = int(meta["n"]), int(meta["n_major"]), int(meta["r"])
        red = 100.0 * (n - s - r_used) / n
        print(f"\n  tau={tau:<5g} majors={s:,}  minors={n-s:,}  reduction={red:.1f}%",
              flush=True)
        if red < 88.0:
            print(f"    reduction below 88% -- exp05 already showed this regime fails; "
                  f"skipping the solve", flush=True)
            rows.append(dict(tau=tau, r=a.rank, majors=s, minors=n - s,
                             reduction_pct=round(red, 1), t_s="", speedup="",
                             gap_pct="", note="skipped: reduction too low"))
            K.write("exp06_regional_tau", rows)
            continue
        t, o, info = K.solve_cpp(d, "hard", a.rank)
        gap = 100.0 * (o - min(o, O_FULL)) / min(o, O_FULL)
        print(f"    t={t:8.1f}s  SPEEDUP={T_FULL/t:5.2f}x  gap={gap:+.4f}%  "
              f"inner={info.get('inner_iters')}", flush=True)
        rows.append(dict(tau=tau, r=a.rank, majors=s, minors=n - s,
                         reduction_pct=round(red, 1), t_s=round(t, 1),
                         speedup=round(T_FULL / t, 2), gap_pct=round(gap, 4),
                         inner=info.get("inner_iters"), obj=o,
                         note="baseline 1085.3s from exp05 (cross-run)"))
        K.write("exp06_regional_tau", rows)

    print("\n[exp06] SUMMARY")
    for x in rows:
        tail = "skipped" if not x["t_s"] else f"{x['speedup']:.2f}x  gap={x['gap_pct']:+.4f}%"
        print(f"  tau={x['tau']:<5g} red={x['reduction_pct']:5.1f}%  {tail}")
    print(f"wrote {C.RESULTS/'exp06_regional_tau.csv'}\n[exp06] DONE", flush=True)


if __name__ == "__main__":
    main()
