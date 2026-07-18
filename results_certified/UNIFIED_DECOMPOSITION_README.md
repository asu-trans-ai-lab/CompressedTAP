# Unified R/C/RC decomposition (consistent framework, July 2026)

`MASTER_UNIFIED_DECOMP.csv` re-runs every case through ONE solver and ONE objective, so the
speedup ratios are consistent from the start. This supersedes the earlier `rc_decomposition_*`
tables, which mixed frameworks (SVD-basis compression, Python vs C++ engines, differing
baselines).

## Framework

All cases solved by the unified column solver on the SAME nonlinear BPR Beckmann objective
`Z = sum_a integral_0^{v_a} t_a(w) dw`, `t_a(v) = t0(1+alpha (v/cap)^beta) + toll`. Three models,
differing ONLY in flow representation (not objective):

- **H0** full-simplex projected gradient (all columns free; demand by exact per-cohort simplex
  projection; NO elimination) -- the baseline.
- **R0** exact per-cohort anchor elimination (reduced gradient).
- **R1** anchor elimination + one fixed nonnegative atom per cohort for the minor set.

`S_R = T_H0/T_R0` (elimination), `S_C|R = T_R0/T_R1` (compression after elimination),
`S_RC = T_H0/T_R1` (combined). Identity `S_R * S_C|R = S_RC` holds for every row. Matched
accuracy: `oe_R1` is the Beckmann objective rel. error of R1 vs H0 (~1e-8 to 1e-6).

## Reading the table

- **Congested real networks are the signal**: Sioux Falls SFK10..120 (multi-OD) give
  S_R ~2-3, S_C|R ~1.2-4.2, S_RC ~2.5-14; Chicago Sketch submitted (Kbar 2.45) S_R ~11.6,
  S_C|R ~1.03 (thin pool -> compression does nothing), S_RC ~13.
- **grid_K* are degenerate here** (uncongested at the given demand -> latent_share 0, gap 0);
  ratios are ~1e-5 s timing noise, not meaningful.
- **Small single-OD cases are timing-noisy** (chicago_one_od runs at sub-0.1 s; S_R wanders
  20-30x between runs). The stable, publication-grade rows are the multi-OD congested networks;
  small cases need larger instances / more repeats.

Reproduce: `column_solver/run_all_decomp.py` (builders convert GMNS+pool -> unified format).
