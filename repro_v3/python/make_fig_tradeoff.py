"""Speed-accuracy trade-off curve across every measured configuration.

x: feasible objective gap (%) -- the accuracy price
y: speedup over the uncompressed solve on the same instance and implementation
Each point is one (instance, rank, threshold, implementation, offset) configuration.
Open markers are the offset-free (w0=0) arm; they are dominated on both axes, which is the
visual form of the load-bearing claim. The dashed line is parity.
"""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fig_tradeoff.pdf"
rd = lambda f: list(csv.DictReader((RES / f).open())) if (RES / f).exists() else []

pts = []  # (gap%, speedup, family, w0?)
for x in rd("grid_axis_cpp.csv"):
    pts.append((float(x["gap_hard_pct"]), float(x["speedup"]), "grid", True))
for x in rd("grid_w0free_cpp.csv"):
    pts.append((float(x["gap_w0free_pct"]), float(x["speedup_w0free"]), "grid", False))
for x in rd("sioux_rank_test.csv"):
    if x["r"]:
        pts.append((float(x["gap_pct"]), float(x["speedup"]), "Sioux Falls", True))
for x in rd("sketch_rank_test.csv"):
    if x["r"]:
        pts.append((float(x["gap_pct"]), float(x["speedup"]), "Chicago Sketch", True))
for f, g in (("kod_axis_alm.csv", "gap_comp_vs_best_pct"),
             ("kod_axis_alm_r20.csv", "gap_comp_vs_best_pct"),
             ("kod_axis_cpp.csv", "gap_hard_vs_best_pct")):
    for x in rd(f):
        pts.append((float(x[g]), float(x["speedup"]), "Chicago Sketch", True))

FL = 5e-4  # floor so exact-zero gaps are plottable on a log axis
fig, ax = plt.subplots(figsize=(6.2, 4.2))
style = {"grid": ("tab:blue", "o"), "Sioux Falls": ("tab:red", "^"),
         "Chicago Sketch": ("tab:green", "s")}
for fam, (c, m) in style.items():
    xs = [max(p[0], FL) for p in pts if p[2] == fam and p[3]]
    ys = [p[1] for p in pts if p[2] == fam and p[3]]
    ax.scatter(xs, ys, c=c, marker=m, s=34, alpha=0.85, edgecolors="none",
               label=f"{fam} (with $w_0$)")
xs = [max(p[0], FL) for p in pts if not p[3]]
ys = [p[1] for p in pts if not p[3]]
ax.scatter(xs, ys, facecolors="none", edgecolors="0.45", marker="o", s=34, lw=0.9,
           label="offset-free ($w_0=0$)")

ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
ax.axvspan(FL, 0.1, color="0.9", zorder=0)
ax.text(1.2e-3, 0.42, "accuracy cost below 0.1%", fontsize=7.5, color="0.35")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("feasible objective gap (\%)  --  accuracy price")
ax.set_ylabel("speedup over uncompressed  ($\times$)")
ax.set_yticks([0.3, 0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0.3","0.5","1","2","5","10"])
ax.set_title("Speed--accuracy trade-off across all measured configurations", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
ax.grid(alpha=0.3, lw=0.5, which="both")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")

best = max(pts, key=lambda p: p[1])
cheap = [p for p in pts if p[3] and p[0] <= 0.1 and p[1] > 1]
print(f"points={len(pts)}  best={best[1]:.2f}x at gap {best[0]:.4f}% ({best[2]})")
print(f"configs with speedup>1 AND gap<=0.1%: {len(cheap)}; "
      f"max among them {max(p[1] for p in cheap):.2f}x")
print("wrote", OUT)
