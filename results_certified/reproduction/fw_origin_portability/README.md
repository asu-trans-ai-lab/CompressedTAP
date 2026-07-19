# Backing records for manuscript Section 7 (convexity preservation + solver portability)

Frozen backing for `paper/section7_fw_origin.tex` (added 2026-07-18;
merged with the author's section7_portability draft 2026-07-19 -- atom
geometry, matched-accuracy sweep table, mirror descent, six-way check,
beyond-traffic prototypes). Every number in the section traces to one of
these files.

2026-07-19 additions:

| file | backs |
|---|---|
| `solver_portability.csv` | Sec. 7.5 sweep table, K<=512 columns (the package's RECORDED Linux single-thread run: LPG 1.02/1.17/3.00/7.70, FW 1.00/1.07/0.90/2.46; RC gaps 0.00/1.19/1.78/1.85%; Chicago RC 0.53%, E/RC 95x FW / 4.3x LPG) |
| `portability_ext.csv` | K=2048 extension cell (n_od=2, Windows single-thread 2026-07-19: LPG 9.36x, FW 19.67x, RC gap 1.2508%, C gap 10.02%); mirror-descent agreement (MD vs LPG rel 2.8e-15 / 2.9e-15 at K=32/512); six-way spread-instance check (LPG rel 4.10e-16 in 81 it, MD 3.25e-13 in 588 it, FW tail 7.35e-5 at 40k it vs C++ BB reference) |
| `solver_portability_windows_20260719.csv` | full K<=512 sweep re-run on Windows: deterministic columns reproduce the recorded run exactly; single-repeat ms-scale wall ratios fluctuate (NOT used in the paper table -- documented here as the cross-machine check) |
| `quick_final_FW_controlled.csv`, `quick_final_ORIGIN_controlled.csv` | Sec. 7.5 FW-native paragraph: grouped-atom speedups 1.06/1.45/4.64 (FW) and 1.20/1.57/6.74 (origin-block) over K=8/16/32 at gaps 0.19/0.11/0.045% |
| `quick_final_cpp_kernel_sioux.csv`, `quick_final_cpp_kernel_chicago.csv` | CSR atom kernel 3.2x / 1.8x at K=32 (96 vs 15 rows) |
| `run_portability_ext.py`, `make_portability_figure.py` | the extension driver (K=2048 cell, latent mirror descent, six-way check) and the figure generator for `paper/fig_solver_portability.pdf` |
| `solver_portability_combined.png` | the combined K=8..2048 figure (PNG twin of the PDF) |

Provenance note: wall-clock ratios are machine-dependent. The paper
table uses the package's recorded Linux run for K<=512 and the dated
Windows extension cell for K=2048, labeled as such in the caption.
Deterministic quantities (optima, gaps, iteration counts, residuals)
reproduce across both machines exactly -- including LPG's 81 iterations
and MD's 3.3e-13 on the spread instance.

Original 2026-07-18 records:

| file | backs | source package |
|---|---|---|
| `FW_controlled.csv` | Table: controlled route-rich FW (K=8/16/32 gaps 0.190/0.114/0.045%, speedups 1.18/2.10/5.17x, storage 1.6/3.2/6.4x) + weighted-QR screening (K=16: 0.015% vs 0.308%, 1.53x; K=32: 0.020% vs 0.245%, 1.12x) | compressed_TAP_repro_package_v1_1_FW_origin, fw_origin_extension/results/reference_sioux/ |
| `ORIGIN_controlled.csv` | Table: origin-block conditional gradient (0.190/0.114/0.044%, 1.14/1.73/5.21x) | same |
| `cpp_kernel_sioux.csv`, `cpp_kernel_chicago.csv` | C++17 kernel check (96 full vs 15 grouped rows; 3.10x / 1.68x single-thread) | same |
| `solver_portability.csv` | Sec. 7.5 table: PG E/RC speedups 1.02/1.17/3.00/7.70 and FW 1.00/1.07/0.90/2.46 over K=8/32/128/512; RC gaps 0.00/1.19/1.78/1.85%; C gaps 0.00/2.89/6.26/6.23%; Chicago Sketch K=32 x2 demand: C 149.8%, RC 0.53%, E/RC 95x (FW) / 4.3x (PG) | compressed_TAP_repro_package_v3_1_LPG, repro/fw/results/solver_portability/ |
| `SOLVER_PORTABILITY.md` | protocol + four conclusions (authors' design note) | same |
| `integrated_check_windows_20260718.txt` | five-solver chain (SLSQP = cpp-BB = FW-atoms = FW-full = LPG): spread instance LPG rel 3.7e-14 vs FW-atoms tail 9.8e-5 on identical atoms; INTEGRATED CHECK PASSED. Fresh Windows single-thread run of repro/interface/integrated_check.py (with the v2-style .exe/-static portability fix, no science change) | same |

Caveats stated in the section itself: wall-clock ratios are hardware- and
run-dependent; objective values, representation gaps, atom counts, storage
bytes, and residuals are deterministic. Speedups are within-solver,
matched-accuracy ratios; representation error is reported separately and
never folded into a speedup.
