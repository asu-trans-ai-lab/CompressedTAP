"""Gradient projection on the compressed (signed-SVD) representation — `GP-signed`.

Author decision 1(a) of 2026-07-19: keep v3's own representation and run a projection
operator on it, so the compressed row of Table 2 and the GP column of Panel B are not
produced by the AL operator.

Compressed variables w = (y, z): y = major path flows, z = latent coordinates, with

    A1 y + M z = d_eff            (OD conservation after removing the nominal minor flow)
    y >= 0                        (major nonnegativity, simple bounds)
    w0 + U_r z >= 0               (reconstructed minor nonnegativity -- DENSE)

There is no closed-form projection onto that set: the equality couples every OD through
the shared latent block, and the last constraint is dense. Two consequences, both of
which are the manuscript's own point about the signed decoder:

  * the equality + bound part is projected EXACTLY, by Dykstra alternation between the
    affine set (one prefactorised least-norm correction) and the bound y >= 0;
  * the dense reconstructed constraint is carried by a quadratic penalty, exactly as the
    AL formulation carries it, with the weight held fixed during the projection.

The operator is therefore labelled `GP-signed (dense constraint penalised)` wherever it
is reported, and never presented as an exact projection onto the full compressed set.
"""
from __future__ import annotations

import time

import numpy as np


class AffineBoundProjector:
    """Dykstra projection onto {E w = b, y >= 0} with E = [A1 | M] prefactorised."""

    def __init__(self, A1, M, d_eff, n_major, ridge=1e-10):
        self.E = np.hstack([np.asarray(A1.todense() if hasattr(A1, "todense") else A1),
                            np.asarray(M)])
        self.b = np.asarray(d_eff, dtype=float).flatten()
        self.s = int(n_major)
        EEt = self.E @ self.E.T
        EEt[np.diag_indices_from(EEt)] += ridge
        self.chol = np.linalg.cholesky(EEt)

    def _affine(self, w):
        r = self.E @ w - self.b
        y = np.linalg.solve(self.chol, r)
        lam = np.linalg.solve(self.chol.T, y)
        return w - self.E.T @ lam

    def __call__(self, w, iters=60, tol=1e-12):
        w = np.asarray(w, dtype=float).copy()
        p = np.zeros_like(w)
        q = np.zeros_like(w)
        for _ in range(iters):
            prev = w
            u = self._affine(w + p)
            p = w + p - u
            v = u + q
            v[: self.s] = np.maximum(v[: self.s], 0.0)
            q = u + q - v
            w = v
            if np.max(np.abs(w - prev)) < tol:
                break
        return self._affine(w)          # finish on the equality


def solve_gp_signed(P, C, cert_fn, tol=1e-6, max_iter=20000, max_seconds=600.0,
                    penalty=1e3):
    """Projected gradient on the compressed representation.

    cert_fn(x_full) -> relative duality gap, so this operator stops on exactly the same
    certificate as the full-space operators (matched accuracy, not a private criterion).
    Returns (x_full_raw, seconds, iterations, achieved_certificate).
    """
    B, bpr = P["B"], P["bpr"]
    U, D, Mmat = C["U"], C["D"], C["M"]
    A1, d_eff, v_base = C["A1"], C["d_eff"], C["v_base"]
    x0m, major, minor = C["x0m"], C["major"], C["minor"]
    B1 = C["B1"]
    s = int(major.sum())
    r = C["r"]

    proj = AffineBoundProjector(A1, Mmat, d_eff, s)

    y = np.maximum(P["x0"][major], 1e-3)
    z = np.zeros(r)
    w = np.concatenate([y, z])
    w = proj(w)

    def unpack(wv):
        return wv[:s], wv[s:]

    def full_x(wv):
        yy, zz = unpack(wv)
        x = np.empty(P["n"])
        x[major] = yy
        x[minor] = x0m + U @ zz
        return x

    def grad(wv):
        yy, zz = unpack(wv)
        # D = B2' U has shape (links x rank): link flow uses D @ z, the
        # gradient w.r.t. z uses D' t.  (Both were transposed in the first draft.)
        v = v_base + np.asarray(B1.T @ yy).flatten() + D @ zz
        t = bpr.t(v)
        g_y = np.asarray(B1 @ t).flatten()
        g_z = D.T @ t
        # penalty on the dense reconstructed nonnegativity
        m = x0m + U @ zz
        viol = np.minimum(m, 0.0)
        if np.any(viol):
            g_z = g_z + penalty * (U.T @ viol)
        return np.concatenate([g_y, g_z])

    t0 = time.perf_counter()
    wp = gp = None
    it = 0
    cg = np.inf
    for it in range(1, max_iter + 1):
        g = grad(w)
        cg = cert_fn(full_x(w))
        if cg <= tol or (time.perf_counter() - t0) > max_seconds:
            break
        if wp is None:
            step = 1.0 / max(float(np.max(np.abs(g))), 1e-12)
        else:
            sv, yv = w - wp, g - gp
            sy = float(sv @ yv)
            step = float(sv @ sv) / sy if sy > 1e-30 else 1.0
            if not (1e-14 < step < 1e14):
                step = 1.0
        wp, gp = w, g
        w = proj(w - step * g)
    return full_x(w), time.perf_counter() - t0, it, cg
