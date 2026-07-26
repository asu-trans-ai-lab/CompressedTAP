# Flow-weighted campaign — clean reproduction

Everything the paper's numerical section needs, re-run with the **flow-weighted** compressed
basis (author decision, 2026-07-25). Self-contained: one config, one entry point, one CSV per
experiment.

## What changed, and why it matters

The basis is now the SVD of `diag(sqrt(f_minor)) · B2` rather than of `B2`. The retained
subspace therefore minimises the **flow-weighted** reconstruction loss, so a minor path
carrying 1000 vehicles counts a thousand times more than one carrying 1 — under the plain SVD
they counted the same. This is the accuracy lever.

Both arms are run (`config.BASES`), because the comparison is itself a result. A fast check
on 2026-07-25 (`../git_dev/CompressedTAP/repro_v3/results/weighted_check.csv`) found it is
**not a free improvement**:

| case | rank | unweighted | weighted |
|---|---|---|---|
| Sioux Falls | 10 | 11.50x / gap 3.4388% | 4.63x / gap **3.0396%** |
| Sioux Falls | 50 | 2.39x / gap 3.4386% | 0.91x / gap **2.7203%** |
| Sketch E0 | 20 | 7.54x / gap 0.0234% | 5.24x / gap **0.0201%** |
| Sketch V2 | 10 | 1.30x / gap 0.0287% | **1.60x** / gap 0.0287% |

Weighting buys accuracy where accuracy was worst (Sioux, −0.4 to −0.7 pp) and costs speed
there; on Sketch the accuracy gain is small and the speed effect is mixed. The campaign
reports both arms so the trade is visible rather than assumed.

## Layout

    config.py      instances, thresholds, ranks, tolerances, basis arms -- the only place
                   any parameter is defined
    common.py      load / compress / solve / metrics / export / CSV -- one implementation of
                   each, shared by every driver
    drivers/
      exp01_rank.py       rank sweep (Sioux, Sketch V2, Sketch E0) x {weighted, unweighted}
      exp02_richness.py   route-richness axis on Chicago Sketch, both arms
      exp03_grid.py       grid family (N x K) + exact offset ablation, all C++
    run_all.py     orchestrator
    results/       one CSV per experiment (flushed after every row)
    logs/          one log per experiment

## Run

    cd campaign_weighted
    python run_all.py                 # everything, in order
    python run_all.py exp01           # one experiment
    python drivers/exp01_rank.py      # or a driver directly

Single thread is enforced through `config.env()`. Timings are medians of three on instances
that solve in under about a minute and single runs above that; repeated identical solves on
the large instances have varied by up to 40% under machine load, which bounds the precision
of those ratios and is reported rather than hidden.

## Conventions that keep the numbers honest

- **Speedups are within-implementation.** A Python compressed time is divided into a Python
  uncompressed time on the same instance, never across engines; the ratio depends on the
  relative throughput of dense and sparse kernels.
- **Every objective is compared after the same feasibility conversion** (per-OD Euclidean
  projection), and gaps are measured against the best feasible point found on that instance.
- **Grid pools are always regenerated**, never cached: a stale pool silently corrupts the
  threshold and the nominal flow.
- **Preprocessing is recorded** (`svd_s` in every CSV) so speedups can be reported both
  online-only and fully charged. The nominal flow itself is still obtained by solving the
  full problem — the open item flagged in the referee analysis.

## Not included here

Regional and Philadelphia are not in this campaign; the submitted Regional pool
(879,625 paths) is no longer on file. The operator comparison (ALM / FW / GP / reduced
gradient) and the stale-`w0` experiment are the next additions.
