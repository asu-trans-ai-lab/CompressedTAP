"""Combined portability figure (K = 8..2048) -> PDF for the manuscript.

Reads results/solver_portability/solver_portability.csv (K<=512) and
portability_ext.csv (K=2048), plots the matched-accuracy E/RC speedup
for FW and PG, and writes fig_solver_portability.pdf (+ png).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "results" / "solver_portability"


def load_speedups(path, inst_prefix):
    sp = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["representation"] == "speedup_E_over_RC" and \
                    r["instance"].startswith(inst_prefix):
                k = int(r["instance"].split("_K")[1].split("_")[0])
                sp[(k, r["solver"])] = float(r["time_to_target_s"])
    return sp


def main(dest_pdf):
    sp = load_speedups(OUT / "solver_portability.csv", "controlled")
    sp.update(load_speedups(OUT / "portability_ext.csv", "controlled"))
    ks = sorted({k for k, _ in sp})
    plt.figure(figsize=(6.4, 4.2))
    for solver, mk, lab in (("PG", "s", "L+PG"), ("FW", "o", "L+FW")):
        ys = [sp[(k, solver)] for k in ks if (k, solver) in sp]
        xs = [k for k in ks if (k, solver) in sp]
        plt.plot(xs, ys, marker=mk,
                 label=r"$T(\mathrm{E})/T(\mathrm{RC})$, %s" % lab)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.axhline(1.0, color="k", lw=0.8, ls=":")
    plt.xlabel("route richness $K$ (paths per OD), latent $m=8$")
    plt.ylabel("matched-accuracy speedup")
    plt.title("Representation gain is solver-independent")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "solver_portability_combined.png", dpi=200)
    plt.savefig(dest_pdf)
    print("wrote", dest_pdf)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else OUT / "fig_solver_portability.pdf")
