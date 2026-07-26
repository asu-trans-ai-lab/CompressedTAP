"""Paired-difference figure: the honest replacement for the pooled trade-off scatter.

Every arrow is a CONTROLLED comparison -- same instance, same rank, same threshold, the two
arms solved back to back in one session. Only one thing changes along each arrow, so the
displacement is interpretable in a way that the position of a point in a pooled cloud is not.

  (a) subspace criterion : plain  ->  flow-weighted
  (b) affine offset      : w0 present -> w0 removed   (flow-weighted basis)

The horizontal axis is exact: feasible objective gaps are deterministic and reproduce bit for
bit across runs. The vertical axis is not: identical solves have varied by 1.1x-1.6x typically
(2.8x worst, under load), so the speed noise floor is drawn explicitly and no vertical
displacement smaller than it should be read as an effect.
"""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fig_paired.pdf"
rd = lambda f: list(csv.DictReader((RES / f).open())) if (RES / f).exists() else []
FL = 5e-4                     # floor so exact-zero gaps sit on a log axis
NOISE = 1.45                  # typical worst spread on an identical solve

grid = rd("exp03_grid.csv"); rich = rd("exp02_richness.csv"); rank = rd("exp01_rank.csv")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.6))

# ---------------------------------------------------------------- (a) basis
def arrow(ax, x0, y0, x1, y1, c):
    ax.annotate("", xy=(max(x1, FL), y1), xytext=(max(x0, FL), y0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.1, alpha=0.85,
                                shrinkA=0, shrinkB=0))

pairs = []
for x in grid:
    pass
gk = {(r["N"], r["K"], r["basis"]): r for r in grid}
for (N, Kx) in sorted({(r["N"], r["K"]) for r in grid}, key=lambda t: (int(t[0]), int(t[1]))):
    u, w = gk.get((N, Kx, "unweighted")), gk.get((N, Kx, "weighted"))
    if u and w:
        pairs.append(("grid", float(u["gap_pct"]), float(u["speedup"]),
                      float(w["gap_pct"]), float(w["speedup"])))
rk = {(r["case"], r["basis"]): r for r in rich}
for case in {r["case"] for r in rich}:
    u, w = rk.get((case, "unweighted")), rk.get((case, "weighted"))
    if u and w:
        pairs.append(("Chicago Sketch", float(u["gap_pct"]), float(u["speedup"]),
                      float(w["gap_pct"]), float(w["speedup"])))
ak = {(r["case"], r["r"], r["basis"]): r for r in rank}
for case, rr in sorted({(r["case"], r["r"]) for r in rank}):
    u, w = ak.get((case, rr, "unweighted")), ak.get((case, rr, "weighted"))
    if u and w:
        fam = "Sioux Falls" if case == "sioux" else "Chicago Sketch"
        pairs.append((fam, float(u["gap_pct"]), float(u["speedup"]),
                      float(w["gap_pct"]), float(w["speedup"])))

col = {"grid": "tab:blue", "Chicago Sketch": "tab:green", "Sioux Falls": "tab:red"}
for fam, x0, y0, x1, y1 in pairs:
    arrow(axA, x0, y0, x1, y1, col[fam])
    axA.plot([max(x1, FL)], [y1], marker="o", ms=3.5, color=col[fam], zorder=3)
for fam, c in col.items():
    axA.plot([], [], color=c, marker="o", ms=4, lw=1.1, label=fam)
axA.set_title("(a) plain $\\rightarrow$ flow-weighted subspace", fontsize=10)

# ---------------------------------------------------------------- (b) offset
# Baseline is the representation WITHOUT the offset; the arrow shows what adding w0 buys,
# so both panels read the same way: from the plain object to the one we advocate.
for r in grid:
    if r["basis"] != "weighted":
        continue
    x0, y0 = float(r["gap_nooffset_pct"]), float(r["speedup_nooffset"])
    x1, y1 = float(r["gap_pct"]), float(r["speedup"])
    axB.plot([max(x0, FL)], [y0], marker="o", ms=3.5, mfc="none", mec="0.45", zorder=3)
    arrow(axB, x0, y0, x1, y1, "tab:purple")
    axB.plot([max(x1, FL)], [y1], marker="o", ms=3.5, color="tab:purple", zorder=3)
axB.plot([], [], color="0.45", marker="o", ms=4, mfc="none", lw=0,
         label="baseline: no offset ($w_0=0$)")
axB.plot([], [], color="tab:purple", marker="o", ms=4, lw=1.1, label="with $w_0$")
axB.set_title("(b) adding the offset: $w_0=0 \\rightarrow w_0$ (weighted basis)", fontsize=10)

for ax in (axA, axB):
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("feasible objective gap (\\%) --- exact")
    ax.set_ylabel("speedup over uncompressed ($\\times$)")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    # explicit speed noise floor
    ax.errorbar([2.2e-3], [6.0], yerr=[[6.0 - 6.0 / NOISE], [6.0 * NOISE - 6.0]],
                fmt="none", ecolor="0.2", capsize=3, lw=1.2)
    ax.text(2.9e-3, 6.0, "speed\nnoise", fontsize=6.8, va="center", color="0.2")

fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")

adv = sum(1 for _, x0, y0, x1, y1 in pairs if y1 > y0 * NOISE)
bet = sum(1 for _, x0, y0, x1, y1 in pairs if x1 < x0)
print(f"(a) basis pairs: {len(pairs)}; gap improves in {bet}; "
      f"speed improves beyond the noise floor in {adv}")
gw = [r for r in grid if r["basis"] == "weighted"]
bet_b = sum(1 for r in gw if float(r["gap_pct"]) < float(r["gap_nooffset_pct"]))
adv_b = sum(1 for r in gw if float(r["speedup"]) > float(r["speedup_nooffset"]) * NOISE)
print(f"(b) offset pairs: {len(gw)}; adding w0 improves the gap in {bet_b}; "
      f"improves speed beyond the noise floor in {adv_b}")
print("wrote", OUT)
