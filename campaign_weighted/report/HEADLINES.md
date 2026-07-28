# Headlines — Compressed Traffic Assignment campaign (2026-07-27)

*One page, plain language. Every number traces to a CSV in `campaign_weighted/results/`;
full equations and tables in `numerical_report_v40.pdf`.*

---

## 1. Chicago Regional: compression now WINS — up to 7.9× faster with BETTER accuracy

All C++, same solver settings both sides, path-richness ladder on the real 39,018-link
Regional network (fixed 20,000 OD pairs):

| paths per OD | full solver | compressed | verdict |
|---:|---:|---:|---|
| 8 (≈ the old failing pool) | 256 s | 199 s | near-tie — matches the old negative |
| 16 | 383 s | 100 s | **3.8× faster AND more accurate** |
| 30 | 875 s | 241 s | **3.6× faster AND more accurate** |
| 45 | 1372 s | **174 s** | **7.9× faster AND more accurate (objective 2.6% better)** |

From 16 paths/OD upward, compression **strictly dominates**: less time, better
equilibrium gap, better objective — and the margin grows with richness.

## 2. Why Regional used to fail — and why we could predict the fix

**Law 1 (size ceiling):** every OD pair must keep one explicit path, so compression is
capped at 1 − 1/(paths per OD). The old Regional pool had 7.5 paths/OD → cap 86.6%,
below the ~90% needed to pay. **No tuning could ever have fixed it** — only a richer
pool could, and when we built one (this ladder), the win appeared exactly as predicted.
The failure was a property of the data, not the method.

## 3. Chicago Sketch: the honest speedup is 1.87× at matched accuracy

Python ladder, both solvers genuinely converging, pre-registered accuracy targets:
compression pays from ~15–25 paths/OD, reaching **1.87× at the tightest reachable
target** on the richest pool (603k paths). Below ~15 paths/OD it loses. This replaces
the old "3.26× with essentially zero accuracy loss" claim, which compared two solvers
that were both stalled at low accuracy.

## 4. Law 2 (accuracy floor) — and the algorithm that breaks it

Plain global-SVD compression cannot get below equilibrium gap ≈ 0.013 **no matter the
rank** (tested 6/10/20; five controlled experiments prove the cause is that one global
mode couples unrelated OD pairs — the basis carries only 1.26% of the needed
per-OD correction, and the rest is blocked by nonnegativity).

**The fix that works:** keep a small global basis, and add per-OD "exchange atoms"
(shift flow from an OD's costliest used route to its cheapest), chosen each iteration
on the ~2,000 worst ODs and **re-priced every iteration as costs move**. Gap drops to
0.004 and keeps falling; statically chosen atoms — any number of them — never break
0.010. Stale atoms, not atoms, were the problem.

## 5. What the paper can now claim, in one sentence

> Compression pays above a measurable route-richness threshold and above a
> representation-accuracy floor — both limits are predictable in advance — and with
> dynamically re-priced per-OD exchange atoms the floor itself is removed; on the
> largest network the compiled solver is then up to 7.9× faster at strictly better
> accuracy.

## 6. Claims retired (do not reuse)

- ~~"3.26× with essentially zero accuracy loss"~~ → moderate-accuracy regime, both sides above gap 0.015.
- ~~"Regional fails"~~ → Regional fails **at 7.5 paths/OD** and strictly wins at 16+.
- ~~"Higher rank fixes accuracy"~~ → the floor is rank-insensitive; the representation class is what matters.

## Reproducibility

Everything is on GitHub (`asu-trans-ai-lab/CompressedTAP`, branch `version_3`):
drivers exp01–exp13, all result CSVs, network + demand data (LFS for large files),
the complete submitted-paper instances, and this report. Verified by a clean clone
from GitHub: files intact (SHA-256 match) and the certified Sioux full-vs-compressed
gap reproduces to 0.02%. Large path pools are not committed — they regenerate
deterministically with documented one-line commands.
