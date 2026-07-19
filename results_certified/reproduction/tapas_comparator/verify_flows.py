"""Verify a TAsK link-flow file against the certified reference.

usage:
    python python/verify_flows.py runs/cs_TAPASf_flows.txt \
        [--ref certified/cs_tapas_flows.txt] \
        [--net Data/ChicagoSketch_net.txt]

TAsK writes one row per link: tail_node head_node flow travel_time.
Flows are compared arc-by-arc on the (tail, head) key; with the TNTP
network file, the Beckmann objective of both flow vectors under the
network's BPR functions is reported as well, so agreement is certified
in the objective and link by link.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def read_flows(path):
    vals = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 3:
            vals[(int(p[0]), int(p[1]))] = float(p[2])
    return vals


def read_tntp(path):
    """{(init, term): (capacity, fftt, b, power)} in file order."""
    out = {}
    started = False
    for line in Path(path).read_text().splitlines():
        if "~" in line:
            started = True
            continue
        if not started:
            continue
        parts = re.split(r"[\s;]+", line.strip())
        if len(parts) >= 8 and parts[0].lstrip("-").isdigit():
            a, bnode = int(parts[0]), int(parts[1])
            out[(a, bnode)] = (float(parts[2]), float(parts[4]),
                               float(parts[5]), float(parts[6]))
    return out


def beckmann(keys, flows, net):
    tot = 0.0
    for k in keys:
        cap, fftt, b, p = net[k]
        v = flows[k]
        ratio = v / cap if cap > 0 else 0.0
        tot += fftt * (v + b * cap * ratio ** (p + 1) / (p + 1))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flows")
    ap.add_argument("--ref", default=str(HERE / "certified"
                                         / "cs_tapas_flows.txt"))
    ap.add_argument("--net", default=str(HERE / "Data"
                                         / "ChicagoSketch_net.txt"))
    args = ap.parse_args()

    fa = read_flows(args.flows)
    fb = read_flows(args.ref)
    if set(fa) != set(fb):
        sys.exit("arc key sets differ (%d vs %d)" % (len(fa), len(fb)))
    keys = sorted(fa)
    va = np.array([fa[k] for k in keys])
    vb = np.array([fb[k] for k in keys])
    d = np.abs(va - vb)
    scale = np.maximum(np.abs(vb), 1.0)
    print("arcs %d | max |dv| %.6g | mean |dv| %.6g | "
          "max rel %.3e | link rel-L2 %.3e"
          % (len(keys), d.max(), d.mean(), (d / scale).max(),
             np.linalg.norm(va - vb) / max(np.linalg.norm(vb), 1e-30)))
    if args.net and Path(args.net).exists():
        net = read_tntp(args.net)
        missing = [k for k in keys if k not in net]
        if missing:
            print("%d arcs missing from net file -- objective skipped"
                  % len(missing))
        else:
            oa = beckmann(keys, fa, net)
            ob = beckmann(keys, fb, net)
            print("Beckmann: candidate %.10e vs certified %.10e "
                  "(rel %.3e)" % (oa, ob, abs(oa - ob) / abs(ob)))


if __name__ == "__main__":
    main()
