# Backing records for manuscript Section 7 (convexity-preserving FW/origin extension)

Frozen backing for `paper/section7_fw_origin.tex` (added 2026-07-18).
Every number in the section traces to one of these files.

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
