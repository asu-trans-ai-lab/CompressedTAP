"""Run the whole flow-weighted campaign, in order, with logs.

    python run_all.py                 # everything
    python run_all.py exp01 exp02     # selected experiments

Each driver writes its own CSV to results/ and flushes after every row, so an interrupted
run loses at most the row in flight. Nothing is cached between experiments except the C++
export directories, which are keyed by (instance, rank, basis).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

DRIVERS = [
    ("exp01", "drivers/exp01_rank.py",     "rank sweep: weighted vs unweighted (Python)"),
    ("exp02", "drivers/exp02_richness.py", "route-richness axis on Chicago Sketch (Python)"),
    ("exp03", "drivers/exp03_grid.py",     "grid family + offset ablation (C++)"),
]
HERE = Path(__file__).resolve().parent


def main():
    want = sys.argv[1:] or [d[0] for d in DRIVERS]
    C.LOGS.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"campaign: basis arms {C.BASES}, headline = {C.BASES[0]}\n", flush=True)
    summary = []
    for tag, rel, desc in DRIVERS:
        if tag not in want:
            continue
        log = C.LOGS / f"{tag}.log"
        print(f"[{tag}] {desc}\n       log -> {log}", flush=True)
        t = time.perf_counter()
        with log.open("w") as fh:
            rc = subprocess.run([sys.executable, "-u", str(HERE / rel)],
                                stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(HERE), env=C.env()).returncode
        el = time.perf_counter() - t
        print(f"[{tag}] rc={rc} in {el/60:.1f} min", flush=True)
        summary.append((tag, rc, el))

    print("\n" + "=" * 72 + "\nCAMPAIGN SUMMARY")
    for tag, rc, el in summary:
        csvp = C.RESULTS / f"{tag}_{dict((d[0], d[1]) for d in DRIVERS)[tag].split('/')[-1][6:-3]}.csv"
        print(f"  {tag}  rc={rc}  {el/60:6.1f} min")
    print(f"  results in {C.RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
