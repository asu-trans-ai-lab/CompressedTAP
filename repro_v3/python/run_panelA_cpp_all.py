"""Autonomous Panel A completion via C++ (regional -> philadelphia -> sketch), then
post-process: recompute Gap_F against the best uncompressed objective and write a
consolidated summary. Unattended; each network is a fresh subprocess so memory is isolated.

    python -u run_panelA_cpp_all.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
LOGS = RESULTS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

NETS = ["regional", "philadelphia", "sketch"]


def run_net(net):
    log = LOGS / f"panelA_cpp_{net}.log"
    print(f"\n{'='*70}\n[cpp-all] {net} -> {log}\n{'='*70}", flush=True)
    t = time.perf_counter()
    with log.open("w") as fh:
        rc = subprocess.run(
            [sys.executable, "-u", str(HERE / "run_table4A_cpp.py"), "--net", net,
             "--rank", "50", "--max-outer", "40", "--max-inner", "300"],
            stdout=fh, stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "OMP_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1"}).returncode
    el = time.perf_counter() - t
    print(f"[cpp-all] {net}: rc={rc} in {el/60:.1f} min", flush=True)
    return rc, el


def recompute(net):
    """Set v^ref to the min feasible objective over all uncompressed rows (tau=0 here IS
    the reference, so this mostly confirms; it also flips any negative Gap_F from a looser
    reference to nonnegative)."""
    p = RESULTS / f"table4A_{net}_cpp.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return None
    # reconstruct each row's absolute feasible objective from stored Gap_F and the common ref
    def frow(r):
        g = float(r["Gap_F_pct"]); of = r.get("obj_feasible")
        if of not in (None, ""):
            return float(of)
        return None
    # obj_feasible column is present (M.evaluate stores it); best uncompressed = tau0 row
    objs = [float(r["obj_feasible"]) for r in rows if r.get("obj_feasible") not in (None, "")]
    tau0 = [float(r["obj_feasible"]) for r in rows
            if r.get("obj_feasible") not in (None, "") and abs(float(r["tau"])) < 1e-12]
    if not tau0:
        return None
    f_best = min(tau0)  # tau=0 IS the reference by decision a
    out = []
    for r in rows:
        r2 = dict(r)
        of = r.get("obj_feasible")
        if of not in (None, ""):
            r2["Gap_F_pct_corrected"] = 100.0 * (float(of) - f_best) / f_best
        r2["f_ref_corrected"] = f_best
        out.append(r2)
    corr = RESULTS / f"table4A_{net}_cpp_corrected.csv"
    keys = sorted({k for r in out for k in r})
    with corr.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    return corr


def main():
    summary = []
    for net in NETS:
        rc, el = run_net(net)
        corr = recompute(net)
        summary.append((net, rc, el, corr))

    print("\n" + "=" * 70 + "\n[cpp-all] SUMMARY", flush=True)
    for net, rc, el, corr in summary:
        p = RESULTS / f"table4A_{net}_cpp.csv"
        nrows = len(list(csv.DictReader(p.open()))) if p.exists() else 0
        print(f"  {net:14s} rc={rc} {el/60:6.1f}min rows={nrows} "
              f"corrected={'yes' if corr else 'no'}", flush=True)
    # consolidated print of every corrected row
    print("\n[cpp-all] all Panel-A rows (C++), corrected Gap_F:")
    print(f"  {'net':12s} {'tau':>5} {'red%':>6} {'delta_F%':>9} {'Gap_F%':>9} "
          f"{'GapFcorr%':>10} {'R2':>7} {'cpu_s':>8}")
    for net in NETS:
        p = RESULTS / f"table4A_{net}_cpp_corrected.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            gc = r.get("Gap_F_pct_corrected", "")
            print(f"  {net:12s} {r['tau']:>5} {float(r['reduction_pct']):>6.1f} "
                  f"{float(r['delta_F_pct']):>9.4f} {float(r['Gap_F_pct']):>9.4f} "
                  f"{(float(gc) if gc else 0):>10.4f} {float(r['link_R2']):>7.4f} "
                  f"{float(r['cpu_s']):>8.1f}")
    print("\n[cpp-all] DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
