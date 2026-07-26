"""Priority 3: is the implemented flow-weighted basis the one we claim?

CLAIM in the docstring: "the SVD is taken of diag(sqrt(f)) B2, so the retained subspace
minimizes the flow-weighted reconstruction loss sum_p f_p ||.||^2".

Let W = diag(sqrt(w)) and let U~ be the leading left singular vectors of W B2.
Eckart-Young gives the best rank-r approximation of W B2 as U~ U~' (W B2), and

    ||W B2 - U~ U~' W B2||  =  ||W ( B2 - W^{-1} U~ U~' W B2 )||,

so the subspace of PATH space that minimises the flow-weighted loss is span(W^{-1} U~),
not span(U~). The implementation uses U~ directly as the path-flow basis. This script
measures, for each candidate basis Phi, the achievable weighted loss

    min_Z || W (B2 - Phi Z) ||_F

and reports which basis actually minimises the criterion we claim to minimise.
"""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C
sys.path.insert(0, str(C.CERTPY))
import compressed_assignment as ca


def weighted_loss(W, B2, Phi):
    """min_Z ||W(B2 - Phi Z)||_F  -- least squares in the W inner product."""
    WB = W @ B2                      # (nm x m)
    WP = W @ Phi                     # (nm x r)
    Z, *_ = np.linalg.lstsq(WP, np.asarray(WB.todense() if sp.issparse(WB) else WB),
                            rcond=None)
    R = np.asarray(WB.todense() if sp.issparse(WB) else WB) - WP @ Z
    return float(np.linalg.norm(R, "fro"))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "sioux"
    r = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    d, pool, tau, _ = C.ROAD[name]
    P = ca.load_problem(str(d), pool, multi_path_only=True)
    major = ca.split_major_minor(P, tau=tau)
    minor = ~major
    B2 = P["B"][minor].astype(float)
    w = np.maximum(np.asarray(P["x0"])[minor], 0.0)
    nm = B2.shape[0]
    print(f"{name}: minors={nm:,}  r={r}  zero-flow minors={(w == 0).sum():,} "
          f"({100*(w==0).mean():.1f}%)  min>0={w[w>0].min():.3e}  max={w.max():.3e}")

    k = min(r, min(B2.shape) - 1)
    # plain basis
    Up, sp_, _ = svds(B2, k=k); Up = Up[:, np.argsort(sp_)[::-1]]
    # weighted factorisation exactly as implemented
    wgt = np.sqrt(w) + 1e-9
    W = sp.diags(wgt)
    Ut, st, _ = svds((W @ B2), k=k); Ut = Ut[:, np.argsort(st)[::-1]]
    # the un-weighted (correct) mapping back to path coordinates
    Uc = Ut / wgt[:, None]
    Uc, _ = np.linalg.qr(Uc)         # orthonormalise; span is what matters

    cands = {
        "plain U (unweighted SVD)": Up,
        "U~ used directly  [AS IMPLEMENTED]": Ut,
        "W^-1 U~ = Phi_r   [CRITERION]": Uc,
    }
    print(f"\n{'basis':38s} {'weighted loss':>14}  {'vs best':>8}")
    losses = {kk: weighted_loss(W, B2, V) for kk, V in cands.items()}
    best = min(losses.values())
    for kk, v in losses.items():
        print(f"{kk:38s} {v:14.4f}  {v/best:7.3f}x")

    # how different are the two subspaces?
    ang = np.linalg.svd(Ut.T @ Uc, compute_uv=False)
    ang = np.clip(ang, -1, 1)
    print(f"\nprincipal angles between span(U~) and span(W^-1 U~): "
          f"min cos={ang.min():.4f}, max cos={ang.max():.4f} "
          f"({'DIFFERENT subspaces' if ang.min() < 0.99 else 'nearly identical'})")


if __name__ == "__main__":
    main()
