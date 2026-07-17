# Certified result tables (revision campaign, July 2026)

Frozen CSV records backing the revised experimental section. Every number in the
manuscript's speedup/accuracy tables traces to one of these files; none were produced by
the code in this repository alone — they come from the authors' research harness (frozen
path-pool instances, sha256-manifested, single-thread timings) with this repository's
`compressed_tap.py` as the base implementation under test.

## Reading rules (they matter)

1. **Timings**: single process, single thread (`OMP_NUM_THREADS=1`), 5 independent process
   invocations per cell; medians with min–max ranges. Runs refuse to start under concurrent
   machine load.
2. **Matched accuracy or no ratio**: a speedup is reported only at an accuracy both models
   reach (`eps_f` column = the floor-matched target). Cells `n/a` mean a model could not
   reach the common floor — the absence is a finding, not a gap in the table.
3. **Objectives are feasible**: every reported objective is evaluated after exact per-OD
   simplex projection. A negative "reference objective difference" would mean an
   under-converged reference and is never reported as favorable.
4. **Baselines are labeled**. `S_R`, `S_C`, `S_CR`, `S_RC` ratios are measured against the
   same-engine ALM full model (H0). The e4 tables additionally carry `gp_ref` — a classical
   path-based gradient-projection solve of the same frozen pool to gap <= 1e-7 — as the
   *accuracy anchor*. Compression speedups are engine- and baseline-dependent (see
   MASTER_DECOMPOSITION_SUMMARY notes column); quoting a ratio without its baseline label
   misstates the result.

## Files

| file | contents |
|---|---|
| `certifications.csv` | the 13 protocol gates (parity, closed-form, fixtures) — all PASS |
| `MASTER_DECOMPOSITION_SUMMARY.csv` | R/C/RC decomposition across grid, Sioux Falls ladders (K/OD 6.7–42.5, incl. flow-weighted), Chicago Sketch, atom family |
| `rc_decomposition_*.csv` | per-instance 2x2 detail behind the master summary |
| `grid_rc_decomposition*.csv` | C++-engine grid two-factor and atom-family runs |
| `e4_03_chicago_sketch_K10.csv` (+manifest) | Chicago Sketch K10 rung (903,516 paths / 81,435 ODs): gp_ref / exact-elimination / compressed, 5 invocations |
| `exp1_weighted_grid.csv` | controlled-grid mechanism study: flow-weighted vs unweighted SVD, EYM bound |
| `exp2_sioux_ladder.csv` | Sioux Falls 4-method benefit ladder (full / major-only / unweighted / weighted) |

Instance freezing: each timed table's instance is identified by sha256 in its manifest;
the Chicago Sketch submitted instance in `data/chicago_sketch/` reproduces the manuscript's
Table-1 statistics (42,774 paths / 17,464 ODs) exactly.
