"""Complete the K=2.45 operator table: the RG-anchor row (the driver was terminated while
GP-signed was stuck in its dense ell x ell projection -- see log). Re-records the five
completed rows verbatim from results/logs/sketch_operators.log and runs RG-anchor live.

GP-signed entry: DNS (does not scale) -- AffineBoundProjector builds a dense
(ell x (s+r)) E and Cholesky of E E' (ell=17,464): >25 CPU-min, 7.3 GB, zero certified
iterations before termination. Architectural, not tuning.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "vendor"))
import v3_metrics as M
import compressed_assignment as ca
from run_table2_consistency import LMOIndex
from run_anchor_reduced import build_anchor, solve_R

P = ca.load_problem(str(ROOT / "data/submitted_v2/sketch"), "pool.csv",
                    multi_path_only=True)
major = ca.split_major_minor(P, tau=4.54)
C = ca.build_compressed(P, major, 50)
lmo = LMOIndex(P["p2od"], P["n_od"], P["n"])

def relgap(x):
    v = np.asarray(P["B"].T @ x).flatten()
    c = np.asarray(P["B"] @ P["bpr"].t(v)).flatten()
    return float(c @ (x - lmo(c, P["d"]))) / max(abs(float(c @ x)), 1e-12)

print("[rg] solving RG-anchor (tau=4.54, r=50)", flush=True)
R = build_anchor(P, C)
t = time.perf_counter(); rr = solve_R(P, C, R, tol=1e-4, max_outer=30)
secs = time.perf_counter() - t
xf = M.convert_euclid(np.asarray(rr["x"], float), P["d"], P["p2od"])
obj = float(P["bpr"].beckmann(M.link_flow(P, xf)))
print(f"  RG-anchor t={secs:.2f}s obj={obj:,.2f} cert={relgap(xf):.3e} "
      f"anchors_active={rr['anchor_active']}", flush=True)

# assemble the table: five rows transcribed from the terminated driver's log + RG live
rows = [
    ("ALM",       "full",       21.60, 16761371.28, 4.393e-04, ""),
    ("GP",        "full",       21.09, 16760323.13, 9.118e-05, "it=408"),
    ("FW",        "full",        1.43, 16760457.82, 9.942e-05, "it=169"),
    ("ALM",       "compressed", 23.20, 16766096.28, 1.621e-03, ""),
    ("GP-signed", "compressed", float("nan"), float("nan"), float("nan"),
     "DNS: dense ell x ell projection (ell=17,464); >25 CPU-min, 0 certified iters"),
    ("RG-anchor", "compressed", round(secs, 2), obj, relgap(xf),
     f"anchors_active={rr['anchor_active']}"),
]
f_best = min(r[3] for r in rows if r[3] == r[3])
tfull = {r[0]: r[2] for r in rows if r[1] == "full"}
import csv
out = HERE.parent / "results" / "sketch_operators.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["op", "space", "time_s", "speedup_vs_full", "obj_feasible",
                "gap_vs_best_pct", "cert_relgap", "note", "source"])
    for op, sp, t_, o_, c_, note in rows:
        base = {"ALM": "ALM", "GP-signed": "GP", "RG-anchor": "GP"}.get(op)
        su = round(tfull[base] / t_, 2) if sp == "compressed" and base and t_ == t_ else ""
        gap = 100 * (o_ - f_best) / f_best if o_ == o_ else ""
        w.writerow([op, sp, t_, su, o_, gap, c_, note,
                    "log" if op not in ("RG-anchor",) or sp == "full" else "live"])
print(f"[rg] wrote {out}", flush=True)
print(f"\nSUMMARY K=2.45 (f_best={f_best:,.2f}):")
for op, sp, t_, o_, c_, note in rows:
    gap = f"{100*(o_-f_best)/f_best:+.4f}%" if o_ == o_ else "  n/a"
    print(f"  {op:10s} {sp:10s} t={t_ if t_==t_ else float('nan'):>7} obj_gap={gap} {note}")
