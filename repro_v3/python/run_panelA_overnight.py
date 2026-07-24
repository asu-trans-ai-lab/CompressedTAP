"""Unattended overnight run of Table 4 Panel A across all networks (decision 3).

Runs the networks large-first (they set the critical path) so a partial night still
produces the hardest cells:

    regional -> philadelphia -> sketch -> sioux

Each network is a separate subprocess (fresh interpreter, isolated memory: Regional alone
loads ~4 GB), single-thread, with the unified reference cap (decision 2) and the
FW-skip-on-large-networks rule (decision 1) already baked into run_table4A_thresholds.py.
A per-network wall-clock ceiling stops a pathological network from eating the whole night.
Every network's stdout is tee'd to its own log and the exit status is recorded.

    python run_panelA_overnight.py [--ref-cap 900] [--solve-cap 1800]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

ORDER = ["regional", "philadelphia", "sketch", "sioux"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-cap", type=float, default=900.0)
    ap.add_argument("--solve-cap", type=float, default=1800.0)
    ap.add_argument("--net-ceiling", type=float, default=9000.0,
                    help="hard wall-clock ceiling per network (s)")
    ap.add_argument("--nets", nargs="*", default=ORDER)
    a = ap.parse_args()

    summary = []
    for net in a.nets:
        log = LOGS / f"panelA_{net}.log"
        print(f"\n{'='*70}\n[overnight] {net} -> {log}\n{'='*70}", flush=True)
        cmd = [sys.executable, str(HERE / "run_table4A_thresholds.py"),
               "--net", net, "--ref-cap", str(a.ref_cap),
               "--solve-cap", str(a.solve_cap)]
        t = time.perf_counter()
        with log.open("w") as fh:
            try:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    timeout=a.net_ceiling,
                                    env={"OMP_NUM_THREADS": "1",
                                         "MKL_NUM_THREADS": "1",
                                         "PATH": __import__("os").environ.get("PATH", "")}
                                    ).returncode
                status = "ok" if rc == 0 else f"exit={rc}"
            except subprocess.TimeoutExpired:
                status = f"TIMEOUT>{a.net_ceiling:.0f}s"
        el = time.perf_counter() - t
        summary.append((net, status, el))
        print(f"[overnight] {net}: {status} in {el/60:.1f} min", flush=True)

    print("\n" + "=" * 70 + "\n[overnight] SUMMARY")
    for net, status, el in summary:
        print(f"  {net:14s} {status:16s} {el/60:7.1f} min")
    csvs = sorted((HERE.parent / "results").glob("table4A_*.csv"))
    print(f"[overnight] CSVs written: {[c.name for c in csvs]}")


if __name__ == "__main__":
    raise SystemExit(main())
