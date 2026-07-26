"""3D view of the grid family: the accuracy surface over (grid size, pool depth).

Accuracy is the axis we can actually resolve --- feasible objective gaps are deterministic and
reproduce bit for bit, whereas identical solves vary by up to ~1.45x in wall time. So the
surface plots the GAP over the (N, K) design, one sheet per subspace criterion, and the
vertical separation between the sheets is the effect of flow weighting.

Panel (b) shows the same design for the offset ablation: the sheet lifts everywhere when w0
is removed, which is the load-bearing result in surface form.
"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fig_3d.pdf"
rows = list(csv.DictReader((HERE / "results" / "exp03_grid.csv").open()))

Ns = sorted({int(r["N"]) for r in rows})
Ks = sorted({int(r["K"]) for r in rows})
KI = np.arange(len(Ks))
X, Y = np.meshgrid(KI, Ns)

def surf(basis, field):
    Z = np.full(X.shape, np.nan)
    for i, N in enumerate(Ns):
        for j, K in enumerate(Ks):
            m = [r for r in rows if int(r["N"]) == N and int(r["K"]) == K
                 and r["basis"] == basis]
            if m:
                Z[i, j] = float(m[0][field])
    return Z

fig = plt.figure(figsize=(12.4, 5.2))

# ---- (a) basis effect on accuracy
ax = fig.add_subplot(1, 2, 1, projection="3d")
Zu, Zw = surf("unweighted", "gap_pct"), surf("weighted", "gap_pct")
ax.plot_surface(X, Y, Zu, alpha=0.55, color="tab:orange", edgecolor="0.5", lw=0.3)
ax.plot_surface(X, Y, Zw, alpha=0.85, color="tab:blue", edgecolor="0.3", lw=0.3)
ax.set_title("(a) accuracy: plain (orange) vs flow-weighted (blue)", fontsize=10, pad=0)
ax.set_zlabel("feasible gap (\\%)", fontsize=8)

# ---- (b) offset effect on accuracy, weighted basis
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
Zw2 = surf("weighted", "gap_pct")
Zn = surf("weighted", "gap_nooffset_pct")
ax2.plot_surface(X, Y, Zn, alpha=0.55, color="tab:red", edgecolor="0.5", lw=0.3)
ax2.plot_surface(X, Y, Zw2, alpha=0.85, color="tab:blue", edgecolor="0.3", lw=0.3)
ax2.set_title("(b) accuracy: $w_0$ removed (red) vs present (blue)", fontsize=10, pad=0)
ax2.set_zlabel("feasible gap (\\%)", fontsize=8)

for a in (ax, ax2):
    a.set_xticks(KI); a.set_xticklabels([f"{k}" for k in Ks], fontsize=8)
    a.set_yticks(Ns); a.set_yticklabels([f"{n}" for n in Ns], fontsize=8)
    a.set_xlabel("pool depth $K$", fontsize=8, labelpad=2)
    a.set_ylabel("grid size $N$", fontsize=8, labelpad=2)
    a.view_init(elev=22, azim=-128)
    a.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
fig.savefig(str(OUT).replace(".pdf", ".png"), bbox_inches="tight", dpi=110)

better = np.nansum(Zw < Zu); total = np.sum(~np.isnan(Zu))
lift = np.nansum(Zn > Zw2)
print(f"accuracy: weighted sheet below plain in {int(better)}/{int(total)} cells")
print(f"offset:   no-w0 sheet above w0 sheet in {int(lift)}/{int(total)} cells")
print(f"mean gap  plain {np.nanmean(Zu):.3f}%  weighted {np.nanmean(Zw):.3f}%  "
      f"no-offset {np.nanmean(Zn):.3f}%")
print("wrote", OUT)
