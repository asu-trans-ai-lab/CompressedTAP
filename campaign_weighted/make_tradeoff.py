"""The accuracy-vs-CPU-time trade-off, from the exp07 HIST_CSV traces.

A single speedup ratio cannot distinguish "converges faster" from "the baseline stalled with
the clock running". The baseline trace on Chicago Sketch K10 reaches within 1.3% of its final
pool_gap in 29 s and then spends 267 s more, so ANY ratio quoted against its 298 s wall time is
dominated by the stall. The honest object is the whole curve: relative gap against elapsed CPU
time, per method, plus the time each method needs to reach a common ladder of accuracy targets.

Reads   <TMP>/reuse_cpp/<pool>/hist_*.csv   (elapsed_s, pool_gap; written by the solver)
Writes  results/tradeoff_<pool>.csv         time-to-target per method, and the floor reached
        figs/tradeoff_<pool>.png            (a) gap vs time  (b) time-to-target vs target

Two conventions, both stated on the figure:
  * the curve plotted is the RUNNING MINIMUM of pool_gap (best answer available at time t).
    The raw trace is non-monotone -- it oscillates by ~20% early on -- and a solver can always
    be stopped at its best iterate, so best-so-far is the meaningful accuracy at time t.
  * trace time is rescaled to the measured wall time, since the trace clock excludes setup.

    python make_tradeoff.py [--pool K10]
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

import config as C

SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "reuse_cpp"
METHODS = ["full-cold", "full-warm", "compressed-reuse"]
STYLE = {"full-cold": ("#444444", "-", "uncompressed, cold"),
         "full-warm": ("#1f77b4", "--", "uncompressed, warm-started"),
         "compressed-reuse": ("#d62728", "-", "compressed, reused basis")}


def envelope(path, wall):
    """(t, best-so-far gap) with the trace clock rescaled to measured wall time."""
    if not Path(path).exists():
        return None
    try:
        H = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except Exception:
        return None
    H = H[np.isfinite(H).all(axis=1)]
    if H.size == 0:
        return None
    t, g = H[:, 0], np.minimum.accumulate(H[:, 1])
    if wall and t[-1] > 0:
        t = t * (wall / t[-1])
    return t, g


def t_to(tr, target):
    """First time the running-min curve is at or below target; None if never."""
    if tr is None:
        return None
    t, g = tr
    k = np.where(g <= target)[0]
    return float(t[k[0]]) if k.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="K10")
    a = ap.parse_args()
    root = SCRATCH / a.pool

    # wall times from the experiment record
    wall = {}
    res = C.RESULTS / "exp07_reuse_cpp.csv"
    if res.exists():
        for r in csv.DictReader(open(res)):
            if r.get("pool") == a.pool and r.get("t_s"):
                wall[(r.get("scenario"), r.get("method"))] = float(r["t_s"])

    scenarios = sorted({p.stem.split("_")[1] for p in root.glob("hist_*_*.csv")})
    if not scenarios:
        raise SystemExit(f"no scenario traces yet in {root}")
    print(f"scenarios: {', '.join(scenarios)}")

    traces = {}
    for sc in scenarios:
        for m in METHODS:
            traces[(sc, m)] = envelope(root / f"hist_{sc}_{m}.csv", wall.get((sc, m)))

    # accuracy ladder spanning what the methods actually achieve
    fl = [tr[1][-1] for tr in traces.values() if tr is not None]
    if not fl:
        raise SystemExit("no usable traces")
    lo, hi = min(fl), 3.5e-2
    ladder = np.geomspace(hi, max(lo, 1e-6), 12)

    rows = []
    for sc in scenarios:
        for m in METHODS:
            tr = traces[(sc, m)]
            if tr is None:
                continue
            d = dict(pool=a.pool, scenario=sc, method=m,
                     wall_s=round(float(tr[0][-1]), 1),
                     floor_gap=float(tr[1][-1]),
                     t_to_1p3x_floor_s=None)
            # how much time is spent AFTER reaching 1.3% of the final gap -- the stall
            near = t_to(tr, tr[1][-1] * 1.013)
            if near is not None:
                d["t_to_1p3x_floor_s"] = round(near, 1)
                d["stall_frac_pct"] = round(100 * (1 - near / tr[0][-1]), 1)
            for g in ladder:
                d[f"t@{g:.2e}"] = (lambda x: None if x is None else round(x, 1))(t_to(tr, g))
            rows.append(d)

    out = C.RESULTS / f"tradeoff_{a.pool}.csv"
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    print(f"\n{'scenario':8s} {'method':18s} {'wall_s':>8s} {'floor':>11s} "
          f"{'t@floor':>9s} {'stall%':>7s}")
    for r in rows:
        print(f"{r['scenario']:8s} {r['method']:18s} {r['wall_s']:8.1f} "
              f"{r['floor_gap']:11.3e} {str(r.get('t_to_1p3x_floor_s')):>9s} "
              f"{str(r.get('stall_frac_pct')):>7s}")

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(no figure: {e})")
        return
    sc0 = scenarios[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in METHODS:
        tr = traces.get((sc0, m))
        if tr is None:
            continue
        c, ls, lab = STYLE[m]
        ax[0].plot(tr[0], tr[1], ls, color=c, lw=1.8, label=lab)
        ax[0].plot(tr[0][-1], tr[1][-1], "o", color=c, ms=5)
    ax[0].set_xlabel("elapsed CPU time (s, single thread)")
    ax[0].set_ylabel("relative gap (best so far)")
    ax[0].set_yscale("log")
    ax[0].set_title(f"(a) accuracy vs time — Chicago Sketch {a.pool}, scenario {sc0}")
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8)

    for m in METHODS:
        ts, gs = [], []
        for g in ladder:
            v = [t_to(traces.get((sc, m)), g) for sc in scenarios]
            v = [x for x in v if x is not None]
            if len(v) == len([s for s in scenarios if traces.get((s, m))]):
                ts.append(float(np.mean(v)))
                gs.append(g)
        if ts:
            c, ls, lab = STYLE[m]
            ax[1].plot(gs, ts, ls, color=c, marker="o", ms=4, lw=1.8, label=lab)
    ax[1].set_xlabel("accuracy target (relative gap)")
    ax[1].set_ylabel("CPU time to reach it (s, mean over scenarios)")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].invert_xaxis()
    ax[1].set_title("(b) cost of accuracy — lower is better")
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    fig.suptitle("Curves are running minima; a method is only comparable where both reach "
                 "the same target.", fontsize=8, y=0.005)
    fig.tight_layout()
    figs = C.HERE / "figs"
    figs.mkdir(exist_ok=True)
    fig.savefig(figs / f"tradeoff_{a.pool}.png", dpi=180)
    print(f"wrote {figs / f'tradeoff_{a.pool}.png'}")


if __name__ == "__main__":
    main()
