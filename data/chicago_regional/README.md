# Chicago Regional — network and demand data

GMNS-format inputs for the Chicago Regional experiments (exp06 tau sweep, exp13 extreme
ladder). Committed 2026-07-27.

| file | size | storage | content |
|---|---|---|---|
| `node.csv` | 0.4 MB | git | 12,982 nodes / zones |
| `link.csv` | 6.3 MB | **git-lfs** | 39,018 links with BPR (vdf) parameters |
| `demand.csv` | 32 MB | **git-lfs** | full OD demand (270,598+ OD pairs after grouping) |
| `mode_type.csv`, `settings.csv` | 1 KB | git | GMNS auxiliaries |
| `extreme_case_M20000/demand_top20000.csv` | 0.3 MB | git | the exp13 instance: top 20,000 ODs by volume (26.7% of total demand) |

## What is deliberately NOT committed: the path pools (column data)

The pool files are large (25 MB – 1.1 GB) and fully **regenerable, deterministically**,
so they are excluded:

| pool | size | how to regenerate |
|---|---|---|
| `path_pool_E0_baseline.csv` (2.02M paths, Kbar 7.48) | 447 MB | original E0 pipeline (see `repro_v3/`) |
| `path_pool.csv`, `path_pool_E2_full_rich.csv` | 1.1 GB each | E2 pipeline |
| exp13 nested pools `pool_X8/16/32/64` (Kbar 7.96/15.67/29.68/44.54) | 25/53/110/179 MB | one command, below |

Regenerate the exp13 nested pools (deterministic: penalty-KSP is seedless, dedupe and
the 2.5x length filter are order-stable; ~1 h single thread):

```bash
python campaign_weighted/drivers/exp12_gen_pools.py --net regional --M 20000 --K 64 --ratio 2.5
```

Then run the ladder:

```bash
python campaign_weighted/drivers/exp13_regional_cpp.py --net regional --M 20000
```

Results these data back: `campaign_weighted/results/exp13_regional_M20000_cpp.csv`
(strict-domination ladder, wall ratio up to 7.90x at Kbar 44.5) and
`exp13_matched_quality.csv`. See `campaign_weighted/report/numerical_report_v40.pdf`.

Note: `demand_top20000.csv` is the *controlled-richness instance* used by exp13 — it is
a demand SUBSET (26.7% of volume). Full-demand Regional results must use `demand.csv`.
