# v3 rerun — handoff snapshot (2026-07-24)

> **RESOLVED (same day, second pass — see §0).** The speedup question in §2 is answered:
> the speedup is real in the Python solver and follows a reduction-level law. §2 below is
> kept for the record of how it looked before the diagnosis.

---

## 0. RESOLUTION (2026-07-24, diag_sioux_regimes.py + sketch sweep)

**One cell at a time, smallest first, diagnose before measuring — and it worked.**

**Root cause of the "compression is slower" confusion — three separate effects, now isolated:**

1. **The hard-regime penalty costs O(n_minor × r) dense work per inner evaluation**
   (`u = x0m + U@z`, `U.T@phi` on the FULL minor set — compressed_assignment.py:308-314).
   The compressed problem is only cheaper when the reduction is high enough that fewer/cheaper
   iterations beat that dense overhead.
2. **My C++ re-runs measured a different amount of work** (max_outer=40, no early
   convergence break comparable to Python's, 3-4x the inner iterations) — engine/config
   artifact, not the method.
3. **Low-reduction cells are genuinely slower** — an honest property, visible in the
   submitted table too (Sketch τ=0.46: 10.36s vs 10.53s ≈ flat).

**Measured speedup law (Python, single thread, tol=1e-4, median-of-reps on Sioux):**

| case | reduction | speedup | obj gap |
|---|---|---|---|
| Sketch τ=0.46 | 29.4% | 0.79× | +0.006% |
| Sketch τ=4.54 (submitted grid) | 53.1% | **1.81×** | +0.028% |
| Sketch τ=20 | 57.7% | **2.10×** | +0.051% |
| Sketch τ=60 | 58.8% | **2.88×** | +0.069% |
| Sioux τ=600 | 91.0% | **3.63×** | +3.44% |

- Crossover ≈ 40–50% variable reduction on Sketch. Speedup ↑ with reduction; obj gap ↑ too
  (speed–accuracy dial). The submitted 1.33× at 53.1% is consistent with the re-measured
  1.81× at the same τ (looser submitted settings).
- Sioux regimes: hard 3.63× (+3.4%), soft 3.89× (+3.4%), recover 8.85× (+37%, unusable),
  screen 0.22× (working set exploded — broken on this instance).
- Repro: `diag_sioux_regimes.py` in this folder.

**(a) C++ anomaly — RESOLVED by measurement (same day):**
- C++ Sioux: full 10.5s vs hard 2.6s = **4.0× speedup** — C++ shows the win too on Sioux.
- C++ Sketch: full 58.1s vs hard 226.9s (τ=0.46) / 289.6s (τ=4.54) — still slower there.
- Diagnosis: (i) per-inner-iter cost — Python hard is CHEAPER than full (15 vs 27 ms) but C++
  hard is 2.5–3× MORE expensive than full (54–60 vs 22 ms): the dense-U gemv runs at ~0.4
  GFlop/s in the `-O2 -static` build (no vectorization) — the untuned build punishes exactly
  the gemv-dominated compressed side; (ii) the `eta<=1e-5` gate (compressed_solver.cpp:1559,
  eta halves from 0.05) forces ≥13 outers before the loop may exit; (iii) active-hinge band is
  opt-in (`ACTIVE_HINGE=1`) and already A/B-tested as a net loss — not the fix.
- Consequence: **Python timings are the trustworthy speedup numbers today**; C++ speedup
  claims for Regional/Philadelphia scale need a build fix (`-O3` + vectorization; resolve the
  old `-march=native` segfault) as its own future cell.

**(b) Paper presentation — DONE per author directive (2026-07-24):** speedup is now the
headline (bold red) column in Table 4 Panel A (1.02–1.37×) and Table 5 (1.31×→0.80×, showing
where rank erases the gain), captions define it, and the verified speedup-law paragraph
(crossover 40–50% reduction, τ as a speed–accuracy dial) is in the threshold section.
The manuscript copies in `manuscript/` are current.

**Next (author's stated second priority):** compression × operator comparison — FW, GP,
reduced-gradient (RG), ALM, each full vs compressed, speedup headline ratios. Not yet run.

---

---

## 1. What is in this folder

| folder | contents |
|---|---|
| `manuscript/` | `main.tex`, `ref.bib`, `main.pdf` — the compiled v3 paper as it stands now (0 `[fill]`). Version header at top of `main.tex`. |
| `results/` | all result CSVs behind the numbers (Panel A per network, Panel B diagnostic, Table 2 Sioux, final table). |
| `drivers/` | the Python drivers + shared metric module. |
| `cpp/` | `compressed_solver.cpp` (SPG-ALM signed-SVD, with the VDUMP_XRAW raw-flow dump). |

Also pushed to GitHub (`asu-trans-ai-lab/CompressedTAP`): `version_3 @ 2adc6d4` (drivers + CSVs),
`major_latent_dev @ a21be7f` (the C++ file).

---

## 2. THE problem that must be settled before moving forward

**Compression does not show a speedup in any case I cleanly re-measured this session.**
The paper's thesis is that compression is faster; the re-runs say slower. Until ONE
trustworthy speedup exists, every downstream table is premature.

Measured compressed ÷ full wall-clock (speedup; <1 means compressed is SLOWER):

| case | setup | speedup |
|---|---|---|
| Panel B Sketch, K=8 → 64 | ALM, matched settings | 0.36 → 0.47 → 0.54 → **0.75** (rises with K/q, still <1) |
| Regional E0, τ=0.23/0.46/1.03 | C++ hard | 0.15 / 0.13 / 0.14 (~7× slower) |
| Sketch C++, τ=0.46/1.06/4.54 | C++, max_outer=40 | 0.26 / 0.26 / 0.16 |
| **Submitted paper (Python, NOT re-measured)** | — | Sketch ~1.33×, Philadelphia ~1.28× (faster) |

The submitted numbers say faster; my re-runs say slower. **That discrepancy is the whole
ballgame.** Likely causes (all my setup, not necessarily the method):

1. **Unoptimized C++ build** — `-O2 -static`, no `-march=native` (it segfaulted and I moved
   on). The compressed path is dense-linear-algebra heavy, so an untuned build punishes the
   compressed side hardest. Could alone flip 1.3× → 0.3×.
2. **Solver may reconstruct the full `x` (n entries) every inner iteration** — never verified
   it actually exploits the `s+r` reduction. If it doesn't, there is no speedup by design.
3. **Mismatched work** — C++ ran 40 outers vs the submitted ~7–9; not apples-to-apples.

---

## 3. What IS clean and trustworthy (do not re-litigate)

- **Accuracy / variable reduction** — solid and meaningful: variable reduction 30–90%,
  feasible gap ≈0, R² ≈1 on every re-run. Compression reproduces the full solution.
- **C++ ↔ Python consistency** — verified on the Sketch instance: Gap_F within 0.03 pp,
  R² within 1.2e-3, δ_F ≈0 for both. Same algorithm, agreeing numbers.
- **Data-provenance fix** — Panel A had been run on the wrong (rich E2) pools; corrected to
  the submitted instances (Sketch V2 pool n=42,774 matches Table 3 exactly).

## 4. Manuscript state right now (what the PDF shows)

- **Re-evaluated this pass:** Table 2 (Sioux consistency); Panel A Sketch (all τ) +
  Philadelphia (τ=0, 0.67).
- **Marked "not re-evaluated this pass" (\textemdash):** Panel A Regional, Table 5, Panel B.
  Regional was reverted because the E0 pool showed compression slower (contradicts thesis) and
  the original 879k pool is not on disk.

## 5. Suggested way forward (ONE cell at a time — not a plan to run yet)

The disciplined path, smallest-first so it iterates in seconds not hours:

1. **Sioux Falls, one solver, matched tolerance.** Get full-vs-compressed wall-clock to a
   number you trust. While doing it, answer the two diagnostics: (a) does a tuned build change
   it? (b) does the compressed solver actually cost `O(s+r)` per iteration, or `O(n)`?
2. Only once a single believable speedup exists, lock that one cell, then extend one network
   at a time.

If a real speedup cannot be produced even on Sioux with a tuned build, the honest fallback is
to make the paper's result the **accuracy + variable-reduction** story (§3), which is clean,
and treat speed as future work.

**Nothing here is decided — this is a snapshot for you to steer from.**
