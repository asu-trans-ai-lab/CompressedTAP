# TAsK_updated — Windows-buildable TAsK (TAPAS et al.) + Python tooling

Curated update of Olga Perederieieva's TAsK solver framework
(github.com/olga-perederieieva/TAsK, MIT license — see
`LICENSE_upstream`) as used in the CompressedTAP project for TAPAS
reference equilibria. Assembled 2026-07-19 from the working copy in
`../TAsK/` (verified file-by-file against a fresh upstream clone).

## What is changed relative to upstream

The C++ **source is unmodified** (`src/` is byte-identical to upstream,
Makefile included). The additions are:

1. `boost_shim/boost/{foreach.hpp, tokenizer.hpp}` — two small header
   shims (BOOST_FOREACH -> C++11 range-for; minimal tokenizer). These
   are the only Boost headers TAsK includes, so the shim removes the
   Boost dependency entirely.
2. `build_windows.sh` — one-command MinGW build: upstream Makefile
   flags (`-Wall -O3 -DUSE_EXTENDED_PRECISION`) plus
   `-std=gnu++11 -I boost_shim -static` (the `-static` link avoids the
   ABI-mismatched-libstdc++-on-PATH failure documented in the
   CompressedTAP repro packages).
3. `params/` — ready-made parameter files for Chicago Sketch and
   Chicago Regional (TAPAS, TAPAS fast, BFW, LUCE, GP, B, PE) from the
   comparator campaign.
4. `python/run_task.py` — drives `task.exe` with a params file,
   redirects `<LINK_FLOWS>` into `runs/` so certified outputs are never
   overwritten, saves the log, reports wall time and final convergence.
5. `python/verify_flows.py` — certifies a produced flow file against
   `certified/cs_tapas_flows.txt` (max/mean/relative link differences +
   Beckmann objective agreement under the TNTP BPR functions).
6. `Data/` — Chicago Sketch and Sioux Falls TNTP inputs. The Chicago
   Regional inputs (65 MB trips) are NOT duplicated; regional params
   reference `../TAsK/Data/` — copy or symlink them here if this folder
   is used standalone.
7. `certified/cs_tapas_flows.txt` — the certified Chicago Sketch TAPAS
   reference (relative gap 3.9e-8), as shipped with the repro packages.

## Quick start

```
sh build_windows.sh                          # -> task.exe
python python/run_task.py params/cs_TAPASf.params
python python/verify_flows.py runs/cs_TAPASf_flows.txt
```

## Status ledger

| item | status |
|---|---|
| Windows build (MinGW g++, static) | see build log next to this file |
| Chicago Sketch TAPAS | certified reference exists (rel gap 3.9e-8); fast-params smoke run verified against it |
| Chicago Regional TAPAS | NOT completed locally: runs reach "Initialisation done" and stall within the session budget (`../TAsK/task_regional*.log`); this is the missing reference blocking the corrected Regional column of the manuscript. Options: long dedicated local run, or a Linux run of this same tree. |
| Upstreaming | src is unmodified, so a branch/PR to the TAsK repository would carry only `boost_shim/`, `build_windows.sh`, and (optionally) the Python tooling — MIT license permits this. Pending author decision. |
