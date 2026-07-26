"""Figure for the rewritten numerical section: the two mechanisms, from committed CSVs.

(a) grid family -- speedup vs candidate-pool depth K, one curve per grid size N
    (grid_axis_cpp.csv, C++, r=20)
(b) rank -- speedup vs r on Sioux Falls in both implementations
    (sioux_rank_test.csv), with the accuracy annotated to show it is flat.
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fig_speedup_mechanisms.pdf"

grid = list(csv.DictReader((RES / "grid_axis_cpp.csv").open()))
rank = list(csv.DictReader((RES / "sioux_rank_test.csv").open()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

Ns = sorted({int(r["N"]) for r in grid})
marks = ["o", "s", "^", "D"]
for i, N in enumerate(Ns):
    rows = sorted((r for r in grid if int(r["N"]) == N), key=lambda r: int(r["K_extra"]))
    ax1.plot([int(r["K_extra"]) for r in rows], [float(r["speedup"]) for r in rows],
             marker=marks[i % 4], lw=1.6, ms=5, label=f"$N={N}$")
ax1.axhline(1.0, color="0.4", ls="--", lw=1.0)
ax1.set_xscale("log", base=2)
ax1.set_xticks([4, 8, 16, 32, 64]); ax1.set_xticklabels([4, 8, 16, 32, 64])
ax1.set_xlabel("candidate paths per OD pair, $K$")
ax1.set_ylabel(r"speedup, compressed / uncompressed")
ax1.set_title("(a) grid family: richer pools, larger gain", fontsize=10)
ax1.legend(fontsize=8, frameon=False, ncol=2)
ax1.grid(alpha=0.3, lw=0.5)

for eng, lab, mk in (("py", "Python", "o"), ("cpp", "C++", "s")):
    rows = sorted((r for r in rank if r["engine"] == eng and r["r"] != ""),
                  key=lambda r: int(r["r"]))
    xs = [int(r["r"]) for r in rows]; ys = [float(r["speedup"]) for r in rows]
    ax2.plot(xs, ys, marker=mk, lw=1.6, ms=5, label=lab)
    for x, y, r_ in zip(xs, ys, rows):
        ax2.annotate(f"{float(r_['gap_pct']):.4f}%", (x, y), textcoords="offset points",
                     xytext=(4, 5), fontsize=6.5, color="0.35")
ax2.axhline(1.0, color="0.4", ls="--", lw=1.0)
ax2.set_xticks([10, 20, 50])
ax2.set_xlabel("compression rank, $r$")
ax2.set_ylabel("speedup, compressed / uncompressed")
ax2.set_title("(b) Sioux Falls: rank moves speed, not accuracy", fontsize=10)
ax2.legend(fontsize=8, frameon=False)
ax2.grid(alpha=0.3, lw=0.5)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
