"""The v3 reference solution `v^ref` — best feasible point from any uncompressed operator.

Author decision A of 2026-07-19, replacing the fixed-operator references of R2/R10.

Why not a fixed operator. Reference quality turned out to be a property of the instance,
not of the solver. On Sioux Falls the AL run terminated 0.048% above the optimum while
gradient projection reached a 7.55e-10 certificate in one second; on Chicago Sketch the
same gradient projection reached only 3.75e-03 in twenty minutes and the AL run found a
better point. Fixing either as `v^ref` produces negative `Gap_F` somewhere in a
three-network study — the exact artifact the v3 revision exists to remove.

What this module does instead. Every uncompressed operator is run under the common
protocol; each terminal iterate is converted to a feasible point by the per-OD simplex
projection (demand equality and nonnegativity together); the feasible point with the
lowest Beckmann objective becomes `v^ref`. `Gap_F >= 0` then holds for every uncompressed
row by construction rather than by luck.

The cost, which belongs in the manuscript and not only here: on large instances the best
available reference is itself only converged to ~1e-3, so **the `Gap_F` column has no
better resolution than that on those networks**. A compressed row whose true gap is
smaller than the reference's own suboptimality cannot be distinguished from zero. Every
CSV therefore carries `ref_source` and `ref_cert` so the reader can see which solution
was used and how well converged it was.
"""
from __future__ import annotations

import time

import numpy as np

import v3_metrics as M


def build_reference(P, ops, tol, verbose=True):
    """Run each uncompressed operator, convert, and keep the best feasible point.

    `ops` maps a label to a callable returning (x_raw, seconds, iterations, certificate);
    a certificate of NaN is allowed for operators that do not report one (e.g. the AL
    method, whose tolerance is a feasibility tolerance rather than an optimality one).

    Returns (ref, candidates) where `ref` carries x_ref / v_ref / f_ref / source / cert.
    """
    cands = []
    for label, fn in ops.items():
        t = time.perf_counter()
        out = fn()
        secs = time.perf_counter() - t
        if isinstance(out, tuple):
            x_raw = out[0]
            iters = out[2] if len(out) > 2 else -1
            cert = out[3] if len(out) > 3 else float("nan")
            secs = out[1] if len(out) > 1 else secs
        else:
            x_raw, iters, cert = out, -1, float("nan")
        xf = M.convert_euclid(x_raw, P["d"], P["p2od"])
        v = M.link_flow(P, xf)
        f = float(P["bpr"].beckmann(v))
        cands.append(dict(label=label, x_raw=x_raw, x_feas=xf, v=v, obj=f,
                          seconds=secs, iterations=iters, cert=cert))
        if verbose:
            print(f"    [ref cand] {label:26s} obj={f:,.6f} cert={cert:.3e} "
                  f"it={iters} t={secs:.1f}s", flush=True)

    best = min(cands, key=lambda c: c["obj"])
    spread = max(c["obj"] for c in cands) - best["obj"]
    if verbose:
        print(f"    -> reference = {best['label']} (obj {best['obj']:,.6f}; "
              f"spread across candidates {spread:.6f} = "
              f"{100 * spread / max(abs(best['obj']), 1e-30):.6f}%)", flush=True)
    ref = dict(x_raw=best["x_raw"], x_ref=best["x_feas"], v_ref=best["v"],
               f_ref=best["obj"], source=best["label"], cert=best["cert"],
               seconds=best["seconds"], iterations=best["iterations"],
               n_candidates=len(cands), obj_spread_pct=
               100.0 * spread / max(abs(best["obj"]), 1e-30))
    return ref, cands
