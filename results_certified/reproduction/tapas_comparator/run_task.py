"""Drive task.exe with a params file; capture the log and key results.

usage:
    python python/run_task.py params/cs_TAPASf.params [--flows-out FILE]

The params file is copied to a temp file with <LINK_FLOWS> redirected to
--flows-out (default: runs/<params-stem>_flows.txt), so the certified
flows in certified/ are never overwritten. The full solver log is saved
to runs/<params-stem>.log, and the summary line reports wall time and
the final convergence value parsed from the log.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--flows-out", default=None)
    ap.add_argument("--exe", default=str(HERE / "task.exe"))
    args = ap.parse_args()

    src = Path(args.params)
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    flows = Path(args.flows_out) if args.flows_out else \
        runs / (src.stem + "_flows.txt")

    text = src.read_text()
    text, n = re.subn(r"<LINK_FLOWS>\s*:\s*\{[^}]*\}",
                      "<LINK_FLOWS>: {%s}" % flows.as_posix(), text)
    if n != 1:
        sys.exit("could not redirect <LINK_FLOWS> in %s" % src)
    tmp = runs / (src.stem + "_run.params")
    tmp.write_text(text)

    log = runs / (src.stem + ".log")
    t0 = time.time()
    with log.open("w") as f:
        r = subprocess.run([args.exe, str(tmp)], cwd=HERE,
                           stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    tail = log.read_text().splitlines()[-25:]
    conv = [l for l in tail if re.search(r"[0-9]e-[0-9]", l)]
    print("exit %d | %.1f s | flows -> %s" % (r.returncode, dt, flows))
    for l in (conv[-3:] if conv else tail[-3:]):
        print("  " + l.strip())
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
