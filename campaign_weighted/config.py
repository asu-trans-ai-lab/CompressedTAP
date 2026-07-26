"""Single source of truth for the flow-weighted campaign.

Every experiment imports from here. Nothing below is duplicated in a driver, so changing an
instance, a threshold or a rank changes it everywhere at once.

BASIS (the 2026-07-25 switch): the compressed basis is now the FLOW-WEIGHTED SVD --- the SVD
of diag(sqrt(f_minor)) B2, so the retained subspace minimises the flow-weighted reconstruction
loss and high-flow minor paths dominate. `BASES` below controls which arms are run; the
unweighted arm is retained so the comparison itself can be reported.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CERTPY = ROOT / "source" / "updated_TAPLite" / "python"
DATA = CERTPY.parent / "data"
V2 = ROOT / "OR_paper_revision_V2" / "m4_rerun" / "data"
CPP = ROOT / "stable_release" / "10_implementation" / "compressed_solver_o3sse.exe"
KSP = ROOT / "stable_release" / "10_implementation" / "ksp_gen.exe"

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"

# ---------------------------------------------------------------- basis arms
WEIGHTED = True                 # campaign default
BASES = ("weighted", "unweighted")   # both arms; first is the headline

# ---------------------------------------------------------------- solver settings
TOL = 1e-4                      # matched stopping tolerance, all operators
MAX_OUTER_FULL = 40
MAX_OUTER_COMP = 30
MAX_INNER = 300
REPS_SMALL = 3                  # medians on instances that solve in < ~60 s
REPS_LARGE = 1                  # single run above that; variance is reported, not hidden

# ---------------------------------------------------------------- road instances
# name -> (directory, pool file, tau, note)
ROAD = {
    "sioux":     (V2 / "sioux",  "pool.csv", 600.0, "real network, 76 links, 91% reduction"),
    "sketch-V2": (V2 / "sketch", "pool.csv",   4.54, "submitted Chicago Sketch pool, Kbar 2.45"),
    "sketch-E0": (DATA / "03_chicago_sketch", "path_pool_E0_baseline.csv", 4.54,
                  "enriched Sketch, Kbar 3.49 -- the sweet spot"),
    "sketch-E2": (DATA / "03_chicago_sketch", "path_pool.csv", 4.54,
                  "rich Sketch, Kbar 8.64"),
    "sketch-K10": (DATA / "03_chicago_sketch", "path_pool_K10.csv", 4.54, "Kbar 11.09"),
    "sketch-K15": (DATA / "03_chicago_sketch", "path_pool_K15.csv", 4.54, "Kbar 15.34"),
}
RICHNESS_AXIS = ["sketch-V2", "sketch-E0", "sketch-E2", "sketch-K10", "sketch-K15"]
RANK_CASES = ["sioux", "sketch-V2", "sketch-E0"]
RANKS = [10, 20, 50]

# ---------------------------------------------------------------- grid family
GRID_NS = [8, 16, 24, 32]
GRID_KS = [4, 8, 16, 32, 64]
GRID_RANK = 20
GRID_TAU_Q = 0.75               # threshold = this quantile of the nominal flow
GRID_Q_PER_OD = 720.0           # demand per OD is GRID_Q_PER_OD / N (holds v/c ~ 1.2)

# operator study runs on these grid cells (where compression is effective)
OPERATOR_CELLS = [(16, 16), (16, 32), (16, 64), (32, 16), (32, 32), (32, 64)]
OPERATOR_CAP_S = 300.0          # wall cap for the slow projection operators


def env():
    import os
    return {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
