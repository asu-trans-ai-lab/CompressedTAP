"""Gate 0 for the v3 campaign: validate v3_metrics.py before any table is produced.

Checks that must pass before a single [fill] cell is computed:
  1. both conversions land in X (Ax = d, x >= 0) to machine tolerance;
  2. Euclidean conversion is the true projection (no feasible point is closer);
  3. an already-feasible x is a fixed point of both maps, giving delta_F = 0;
  4. Gap_F is nonnegative when the reference is the optimum (convexity);
  5. Gap_F = 0 and R^2 = 1 when evaluated at the reference itself;
  6. the two conversions genuinely differ on infeasible input (why we report both).
"""
import numpy as np
import v3_metrics as M


class _BPR:
    def __init__(self, m, seed=0):
        r = np.random.default_rng(seed)
        self.t0 = r.uniform(1.0, 3.0, m)
        self.cap = r.uniform(500.0, 900.0, m)
        self.alpha, self.beta = 0.15, 4.0

    def t(self, v):
        return self.t0 * (1 + self.alpha * (np.maximum(v, 0) / self.cap) ** self.beta)

    def beckmann(self, v):
        v = np.maximum(v, 0)
        return float(np.sum(self.t0 * (v + self.alpha * self.cap
                                       * (v / self.cap) ** (self.beta + 1)
                                       / (self.beta + 1))))


def _toy(n_od=4, k=5, m=12, seed=1):
    r = np.random.default_rng(seed)
    n = n_od * k
    p2od = np.repeat(np.arange(n_od), k)
    A = np.zeros((n_od, n))
    for j, i in enumerate(p2od):
        A[i, j] = 1.0
    B = (r.random((n, m)) < 0.35).astype(float)
    d = r.uniform(500, 900, n_od)
    return {"A": A, "B": B, "d": d, "p2od": p2od, "n_od": n_od,
            "bpr": _BPR(m, seed), "v0": 0.0}, n


def main():
    P, n = _toy()
    rng = np.random.default_rng(7)
    sl = M.od_slice_index(P)

    # a feasible reference
    xref = np.concatenate([P["d"][i] * np.ones(5) / 5 for i in range(P["n_od"])])
    vref = M.link_flow(P, xref)
    fref = float(P["bpr"].beckmann(vref))

    # an infeasible terminal iterate (negatives + wrong demand), as a compressed solve gives
    xt = xref + rng.normal(0, 60, n)

    ok = True

    for conv in ("euclid", "rescale"):
        xf = (M.convert_euclid(xt, P["d"], P["p2od"], sl) if conv == "euclid"
              else M.convert_rescale(xt, P["A"], P["d"], P["p2od"]))
        res = float(np.max(np.abs(P["A"] @ xf - P["d"])))
        neg = float(np.min(xf))
        good = res < 1e-9 and neg >= -1e-12
        ok &= good
        print(f"[1] {conv:8s} lands in X: demand_res={res:.2e} min={neg:.2e} "
              f"{'OK' if good else 'FAIL'}")

    # 2. Euclidean map is the closest feasible point (beat it with random feasible points)
    xe = M.convert_euclid(xt, P["d"], P["p2od"], sl)
    de = np.linalg.norm(xe - xt)
    closer = 0
    for _ in range(2000):
        cand = np.concatenate([
            (lambda w: P["d"][i] * w / w.sum())(rng.random(5)) for i in range(P["n_od"])])
        if np.linalg.norm(cand - xt) < de - 1e-9:
            closer += 1
    ok &= closer == 0
    print(f"[2] Euclidean is the projection: {closer}/2000 random feasible points closer "
          f"{'OK' if closer == 0 else 'FAIL'}")

    # 3. feasible input is a fixed point -> delta_F = 0
    for conv in ("euclid", "rescale"):
        e = M.evaluate(P, xref, fref, vref, conv, sl if conv == "euclid" else None)
        good = e["delta_F_pct"] < 1e-9
        ok &= good
        print(f"[3] {conv:8s} fixed point on feasible x: delta_F={e['delta_F_pct']:.2e}% "
              f"{'OK' if good else 'FAIL'}")

    # 4/5. reference self-evaluation: Gap_F = 0, R^2 = 1
    e = M.evaluate(P, xref, fref, vref, "euclid", sl)
    good = abs(e["Gap_F_pct"]) < 1e-12 and abs(e["link_R2"] - 1.0) < 1e-12
    ok &= good
    print(f"[5] self-evaluation: Gap_F={e['Gap_F_pct']:.2e}% R2={e['link_R2']:.12f} "
          f"{'OK' if good else 'FAIL'}")

    # 4. Gap_F >= 0 requires a TRUE optimum as reference (convexity). Solve the toy to
    #    high accuracy by projected gradient; a random "best-of-N" proxy is not enough.
    xo = xref.copy()
    gprev = xprev = None
    for _ in range(200000):                       # BB-step projected gradient
        g = np.asarray(P["B"] @ P["bpr"].t(M.link_flow(P, xo))).flatten()
        if gprev is None:
            step = 1.0
        else:
            s_, y_ = xo - xprev, g - gprev
            sy = float(s_ @ y_)
            step = float(s_ @ s_) / sy if sy > 1e-30 else 1.0
            if not (1e-12 < step < 1e12):
                step = 1.0
        xprev, gprev = xo, g
        xo_new = M.convert_euclid(xo - step * g, P["d"], P["p2od"], sl)
        if np.max(np.abs(xo_new - xo)) < 1e-13:
            xo = xo_new
            break
        xo = xo_new
    fo = float(P["bpr"].beckmann(M.link_flow(P, xo)))
    vo = M.link_flow(P, xo)
    # optimality check: no feasible descent direction of meaningful size
    gpn = float(np.max(np.abs(M.convert_euclid(
        xo - 1e-3 * np.asarray(P["B"] @ P["bpr"].t(vo)).flatten(),
        P["d"], P["p2od"], sl) - xo)))
    negs = 0
    worst = 0.0
    for _ in range(300):
        xt2 = xo + rng.normal(0, 80, n)
        g = M.evaluate(P, xt2, fo, vo, "euclid", sl)["Gap_F_pct"]
        worst = min(worst, g)
        if g < -1e-9:
            negs += 1
    good = negs == 0
    ok &= good
    print(f"[4] Gap_F >= 0 vs converged optimum (pg-norm {gpn:.2e}): "
          f"{negs}/300 negative, worst {worst:.3e}% {'OK' if good else 'FAIL'}")

    # 6. delta_F is PROVABLY map-independent: per OD, both maps move L1 mass
    #        sum_{x<0}|x| + (sum_{x>=0} x - d)
    #    (Euclidean: k*theta = sum_S x - d and clipped terms contribute |x_j|; rescale:
    #     clip then scale.)  So the P_X ambiguity cannot affect delta_F -- but it does
    #    move x^F, hence Gap_F and R^2.  Verify both halves of that statement.
    xt_neg = xref + rng.normal(0, 400, n)
    frac_neg = float(np.mean(xt_neg < 0))
    b = M.evaluate_both(P, xt_neg, fref, vref, sl)
    d_same = abs(b["delta_F_pct"] - b["rescale_delta_F_pct"]) < 1e-8
    g_diff = abs(b["Gap_F_pct"] - b["rescale_Gap_F_pct"]) > 1e-8
    good = d_same and g_diff
    ok &= good
    print(f"[6] with {100*frac_neg:.0f}% negative entries: "
          f"delta_F identical ({b['delta_F_pct']:.4f}% vs "
          f"{b['rescale_delta_F_pct']:.4f}%) {'OK' if d_same else 'FAIL'}; "
          f"Gap_F differs ({b['Gap_F_pct']:.4f}% vs {b['rescale_Gap_F_pct']:.4f}%) "
          f"{'OK' if g_diff else 'FAIL'}")

    print("\nGATE 0:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
