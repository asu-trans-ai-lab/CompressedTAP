# Warm-start benchmark + scenario break-even (manuscript Sec. 7, tab:warmstart)

One-table experiment answering the reviewers' warm-start challenge with
full preprocessing accounting. Single-thread run of 2026-07-19.

- driver: run_warmstart_benchmark.py (runs inside
  compressed_TAP_repro_package_v3_1_LPG/repro/fw/src/)
- record: warmstart_benchmark.csv (all table cells trace here)

Design: largest dense-prototype instances (Sioux Falls 199 ODs,
Chicago Sketch 400 ODs, K=32, target v/c 2.0); base scenario solved
cold = the nominal equilibrium (T_nominal); T_pool and T_atoms timed
separately; 5-scenario +/-5% per-OD demand stream; every cell LPG at
matched accuracy; warm starts rescale the previous scenario solution
per OD. N_break = (T_nominal+T_pool+T_atoms)/(T_E-warm - T_RC-warm).

Result (honest, negative for compression in this regime): E-warm is
within 1-3 ms/scenario of RC-warm, RC gap 0.00% (anchor-supported),
N_break = 782 (SF) and 11,528 (CS). Compression pays only in the
high-K/m regime of the portability sweep, not on anchor-supported
instances, however many scenarios are streamed. Excluded charges
(stated in Remark rem:nbreak): hidden-path repricing during promotion,
pool updates under material network change.
