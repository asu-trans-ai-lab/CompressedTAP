# repro_v3 — reproduction tree for manuscript v3

Fills every `[fill]` cell in the v3 manuscript from fresh runs. See `spec/TABLES.md` for
the cell-by-cell map, the three governing author decisions, and the gates.

    python python/test_v3_metrics.py        # Gate 0 (must pass first)
    python python/run_table2_consistency.py
    python python/run_table4A_thresholds.py --net {sketch,regional,philadelphia}
    python python/run_table4B_richness.py
    python python/run_table5_rank.py --net {regional,philadelphia}
    python fill_tables.py                   # results/*.csv -> LaTeX rows

Single-thread, quiet machine, median of 3. Python drivers use the certified
`compressed_assignment.py` / `run_progressive_ladder.py`; C++ uses `column_solver`
(ALM / PG / RG) and `tapkernel`.
