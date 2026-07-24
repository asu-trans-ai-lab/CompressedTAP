"""Option 3 of REMARKS R15: recompute Gap_F against the true best uncompressed feasible point.

On the large networks every uncompressed candidate is truncated by a wall-clock cap, so the
reference candidate and the tau=0 threshold row can be differently converged and the tau=0
Gap_F can come out negative. This is a comparison artifact, not compression beating the full
model. The raw objectives are all on disk, so the fix needs no re-solve:

  v^ref := min feasible Beckmann objective over ALL uncompressed solutions in the campaign
           for that network -- i.e. the reference candidates AND the tau=0 threshold row.

Then every Gap_F is recomputed as 100*(obj_row - f_ref_best)/f_ref_best. This can only
lower a positive gap or make a negative one nonnegative; it never inflates compression.

The CSV stores the FEASIBLE objective per row implicitly (obj_feasible when present) and
the reference objective via Gap_F and the stored f_ref. We reconstruct each row's absolute
feasible objective from f_ref*(1+Gap_F/100) and take the campaign-wide min as the corrected
reference. tau=0's own feasible objective is included, which is the whole point.

    python recompute_gapf.py            # all table4A_*.csv in results/
    python recompute_gapf.py --net regional
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def recompute(path: Path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    # absolute feasible objective per row, reconstructed from the stored f_ref and Gap_F
    def f_ref_stored(r):
        # every row shares the same reference objective; back it out where possible
        g = float(r["Gap_F_pct"])
        of = r.get("obj_feasible")
        if of not in (None, ""):
            # obj_feasible is f(v^F); f_ref = obj_feasible / (1 + g/100)
            return float(of) / (1.0 + g / 100.0)
        return None
    # campaign-wide best uncompressed feasible objective:
    #   candidates aren't rows, but the tau=0 row IS an uncompressed feasible solve, and
    #   its obj_feasible is stored; the reference objective is also recoverable per row.
    feas_objs = []
    for r in rows:
        of = r.get("obj_feasible")
        if of not in (None, ""):
            feas_objs.append((float(of), r.get("tau"), r.get("reduction_pct")))
    stored_ref = None
    for r in rows:
        fr = f_ref_stored(r)
        if fr is not None:
            stored_ref = fr
            break
    # true best uncompressed = min(stored reference, tau=0 row's feasible objective)
    tau0 = [float(r["obj_feasible"]) for r in rows
            if r.get("obj_feasible") not in (None, "") and float(r.get("tau", 1)) == 0.0]
    candidates = [x for x in [stored_ref] + tau0 if x is not None]
    if not candidates:
        return None
    f_best = min(candidates)

    out = []
    for r in rows:
        of = r.get("obj_feasible")
        r2 = dict(r)
        if of not in (None, ""):
            r2["Gap_F_pct_corrected"] = 100.0 * (float(of) - f_best) / f_best
        r2["f_ref_corrected"] = f_best
        r2["ref_source_corrected"] = "min(uncompressed candidates + tau0 row)"
        out.append(r2)

    corr = path.with_name(path.stem + "_corrected.csv")
    keys = sorted({k for r in out for k in r})
    with corr.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)

    print(f"\n{path.name}: corrected reference f_best={f_best:,.4f}")
    print(f"  {'tau':>6} {'red%':>6} {'Gap_F (stored)':>16} {'Gap_F (corrected)':>18} "
          f"{'delta_F%':>10}")
    negs = 0
    for r in out:
        gc = r.get("Gap_F_pct_corrected")
        if gc is not None and gc < -1e-6:
            negs += 1
        print(f"  {r.get('tau',''):>6} {float(r.get('reduction_pct',0)):>6.1f} "
              f"{float(r['Gap_F_pct']):>16.5f} "
              f"{(gc if gc is not None else float('nan')):>18.5f} "
              f"{float(r.get('delta_F_pct',0)):>10.4f}")
    print(f"  -> corrected Gate 2: {'PASS' if negs == 0 else f'FAIL ({negs} negative)'}")
    print(f"  wrote {corr.name}")
    return corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None)
    a = ap.parse_args()
    pat = f"table4A_{a.net}.csv" if a.net else "table4A_*.csv"
    files = sorted(RESULTS.glob(pat))
    files = [f for f in files if not f.stem.endswith("_corrected")]
    if not files:
        print(f"no CSVs matching {pat}")
        return
    for f in files:
        recompute(f)


if __name__ == "__main__":
    main()
