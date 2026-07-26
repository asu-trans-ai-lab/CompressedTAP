# Reproducibility, workflow, and results inventory

Every number in the paper traces to a row of a CSV in this repository, and every CSV traces to
one command. This note gives the mapping in both directions, the counts, and the precision
policy that decides which differences the study is entitled to call effects.

---

## 1. Precision policy (read this before any timing comparison)

**Accuracy is exact.** Feasible objective gaps are deterministic. Re-solving the same instance
with the same settings reproduces the objective bit for bit — verified repeatedly, e.g. Chicago
Sketch K15 gave `16,794,251.42` on runs whose wall times were 908 s and 1398 s.

**Timing is not.** Re-measuring *identical* solves across the campaign:

| identical solve | min | max | spread |
|---|---|---|---|
| Sketch-E0 full (Python) | 126.7 s | 205.0 s | 1.62x |
| Sketch-K15 full (Python) | 908.2 s | 1398.2 s | 1.54x |
| Sketch-E2 full (Python) | 486.4 s | 672.7 s | 1.38x |
| Sketch-K10 full (Python) | 645.0 s | 707.9 s | 1.10x |
| Regional full (C++) | 785.3 s | 987.5 s | 1.26x |
| Sketch-V2 full (Python) | 12.2 s | 33.7 s | 2.76x |
| Sioux full (Python) | 1.00 s | 2.36 s | 2.36x |

**Policy: 1.45x is the noise floor.** A speed difference below it is not reported as an effect.
Applying this to the 34 controlled weighted-vs-plain pairs: the gap improves in 27, the speed
exceeds the floor in 2 — which is why the flow-weighting result is stated as an accuracy
improvement with no speed claim.

**Speedups are within-implementation only.** A Python compressed time is divided into a Python
uncompressed time on the same instance; never across engines. The ratio depends on the relative
throughput of dense and sparse kernels, which differs between the two implementations, and that
dependence is itself a reported finding.

---

## 2. Workflow: command -> CSV -> where it appears

### Flow-weighted campaign (`campaign_weighted/`, the current basis)

| command | CSV | feeds |
|---|---|---|
| `python run_all.py exp01` | `results/exp01_rank.csv` | rank subsection; paired figure |
| `python run_all.py exp02` | `results/exp02_richness.csv` | flow-weighted table; paired figure |
| `python run_all.py exp03` | `results/exp03_grid.csv` | grid study; 3D figure; offset ablation |
| `python run_all.py` | all three | full campaign, ~165 min |

Parameters live only in `config.py`; `common.py` holds one implementation each of load /
compress / solve / metrics / export / CSV. No driver defines a parameter or reimplements a
metric.

### Earlier unweighted campaign (`repro_v3/`, retained for comparison)

| command | CSV | feeds |
|---|---|---|
| `run_kod_axis.py --rank 20` | `kod_axis_alm_r20.csv` | Panel B richness axis |
| `run_kod_axis_cpp.py` | `kod_axis_cpp.csv` | engine-dependence discussion |
| `run_sketch_rank_test.py` | `sketch_rank_test.csv` | rank correction |
| `run_sioux_rank_test` (inline) | `sioux_rank_test.csv` | rank table |
| `run_grid_axis_cpp.py` | `grid_axis_cpp.csv` | grid mechanism matrix |
| `run_grid_w0free.py` | `grid_w0free_cpp.csv` | offset ablation |
| `run_grid_verify.py` | `grid_verify.csv` | cross-engine / cross-rank checks |
| `run_grid_operators.py` | `grid_operators.csv` | operator study |
| `run_sketch_operators.py` + `run_rg_row.py` | `sketch_operators.csv` | operator study |
| `run_highk_fixes.py` | `highk_fixes.csv` | falling-branch remedies |
| `run_table2_consistency.py --net sioux --rank 20` | `table2_consistency_sioux_r20.csv` | consistency table |
| `run_table4A_cpp.py --net sketch --rank 20` | `table4A_sketch_cpp.csv` | Panel A |

### Figures

| figure | generator | inputs |
|---|---|---|
| `fig_speedup_mechanisms.pdf` | `repro_v3/python/make_fig_speedup.py` | `grid_axis_cpp.csv`, `sioux_rank_test.csv` |
| `fig_paired.pdf` | `campaign_weighted/make_fig_paired.py` | `exp01`, `exp02`, `exp03` |
| `fig_3d.pdf` | `campaign_weighted/make_fig_3d.py` | `exp03_grid.csv` |

`fig_tradeoff.pdf` was generated and then **withdrawn**: it pooled heterogeneous configurations
on one plane, which presented timing noise as structure. It is superseded by the paired figure.

---

## 3. Results inventory

| file | rows | content |
|---|---|---|
| `exp01_rank.csv` | 18 | rank x basis, Python |
| `exp02_richness.csv` | 10 | richness axis x basis, Python |
| `exp03_grid.csv` | 40 | grid N x K x basis, with offset arm, C++ |
| `grid_axis_cpp.csv` | 20 | grid matrix, plain basis, C++ |
| `grid_w0free_cpp.csv` | 20 | offset ablation, grid |
| `grid_verify.csv` | 14 | cross-engine and cross-rank verification |
| `kod_axis_alm.csv` / `_r20.csv` / `_cpp.csv` | 5 / 3 / 5 | Sketch richness, three configurations |
| `sketch_rank_test.csv` / `sioux_rank_test.csv` | 16 / 8 | rank x engine |
| `sketch_operators.csv` / `grid_operators.csv` | 6 / 36 | operator studies |
| `highk_fixes.csv` | 6 | rank and grouped-representation remedies |
| `table2_consistency_sioux_r20.csv` | 6 | operator consistency |
| `table4A_sketch_cpp.csv` | 4 | Panel A, Chicago Sketch |
| `weighted_check.csv` | 16 | first weighted-vs-plain probe |
| **total** | **233** | recorded configurations |

`exp03` alone represents about 100 solver invocations (each cell shares one uncompressed
baseline across five compressed variants).

**Counts used in the text:** 34 controlled weighted-vs-plain pairs (gap improves in 27, speed
beyond noise in 2); 20 grid cells for the offset ablation (gap rises in 16 when the offset is
removed; weighted sheet below plain in 15).

---

## 4. Environment

- Single thread throughout (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, set centrally in
  `config.env()`).
- C++: `compressed_solver_o3sse.exe`, built `-O3 -funroll-loops -static`. **Not**
  `-march=native` and **no** AVX2 codegen: both segfault in full mode under MinGW. The
  `-O3` baseline-SSE2 binary reproduces the certified `-O2` objectives bit for bit and is
  1.4x / 1.6x faster in full / compressed mode.
- Python: NumPy/SciPy reference implementation, `compressed_assignment.py`.
- Grid pools are regenerated on every run, never cached: a stale pool silently corrupts the
  threshold and the nominal flow, which happened once and produced a table that had to be
  discarded.

## 5. Known gaps

- Timings on the large instances are single runs. Medians of three to five would let
  near-parity entries be ordered; at present they cannot be.
- No reported speedup charges the SVD or the nominal-flow computation. `svd_s` is recorded in
  every CSV so this can be added; the nominal flow is still obtained by solving the full
  problem, which is the open circularity.
- Chicago Regional and Philadelphia are not re-evaluated at the corrected rank. The submitted
  Regional pool (879,625 paths) is no longer on file; the closest available instance has
  2,024,525 paths.
