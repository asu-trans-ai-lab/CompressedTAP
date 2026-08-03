"""Naive global SVD vs block-diagonal (per-OD / per-origin) SVD of the minor
incidence block B2. The paper's compression is w ~ U z with U from a global
truncated SVD of B2; that U is DENSE and its columns couple minor paths across
different OD pairs, so the decoder D = B2' U is dense (every latent coordinate
touches every link). A block-diagonal U -- one block per OD (or per origin) --
keeps each latent coordinate on its own block's links, so D is sparse. We measure
build time, nnz(D), and reconstruction fidelity at matched or better accuracy.

usage: python oblock_svd_compare.py <cpp_problem_dir> [rank] [block=od|origin]
"""
import sys, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

dd = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 50
BLOCK = sys.argv[3] if len(sys.argv) > 3 else 'od'

Bp = np.fromfile(dd + '/B_indptr.i64', dtype=np.int64)
Bi = np.fromfile(dd + '/B_indices.i32', dtype=np.int32)
p2od = np.fromfile(dd + '/p2od.i32', dtype=np.int32)
x0 = np.fromfile(dd + '/x0.f64')
npath = len(Bp) - 1
m = int(Bi.max()) + 1
n_od = int(p2od.max()) + 1
# path x link incidence
indptr = Bp.astype(np.int64)
B = csr_matrix((np.ones(len(Bi)), Bi.astype(np.int64), indptr), shape=(npath, m))
print(f"pool: {npath:,} paths, {m:,} links, {n_od:,} ODs, rank budget r={R}")

# major/minor split on nominal flow (tau=1.0 as in export_problem)
tau = 1.0
major = x0 > tau
minor = ~major
B2 = B[minor]                              # n_minor x m
x0m = x0[minor]
od_minor = p2od[minor]
n_minor = B2.shape[0]
print(f"minor paths: {n_minor:,}  ({100*n_minor/npath:.1f}% of pool)")

true_link = np.asarray(B2.T @ x0m).flatten()   # true minor link flows
den = np.linalg.norm(true_link) + 1e-30

def rsvd(A, k, p=10, q=2, seed=0):
    """Top-k left singular vectors of sparse A (n x m) by randomized range
    finding (Halko-Martinsson-Tropp): O(nnz(A) * (k+p)), fast at scale."""
    rng = np.random.default_rng(seed)
    l = k + p
    Om = rng.standard_normal((A.shape[1], l))
    Y = np.asarray(A @ Om)                       # n x l
    for _ in range(q):
        Y = np.asarray(A @ np.asarray(A.T @ Y))  # power iteration
    Q, _ = np.linalg.qr(Y)                        # n x l
    Bt = np.asarray(A.T @ Q)                      # m x l (sparse @ dense)
    Ub, s, _ = np.linalg.svd(Bt.T, full_matrices=False)   # svd of l x m
    return Q @ Ub[:, :k], s[:k]

def recon_err(U):
    # link-flow error of projecting the nominal minor flow onto span(U).
    # Works for dense or sparse U via matvecs (never densifies U).
    z = np.asarray(U.T @ x0m).flatten()
    approx = np.asarray(B2.T @ np.asarray(U @ z).flatten()).flatten()
    return np.linalg.norm(true_link - approx) / den

# ---------- naive GLOBAL SVD (randomized: tractable at scale) ----------
t0 = time.time()
k = min(R, min(B2.shape) - 1)
Ug, sg = rsvd(B2.astype(float), k)
tg = time.time() - t0
Dg = np.asarray(B2.T @ Ug)                 # m x r dense
nnz_Dg = int((np.abs(Dg) > 1e-9).sum())
print(f"\nGLOBAL SVD  r={k}:  build {tg:6.2f}s  "
      f"nnz(D)={nnz_Dg:,} ({100*nnz_Dg/Dg.size:.1f}% dense)  "
      f"recon_err={recon_err(Ug):.4e}")

# ---------- BLOCK-DIAGONAL SVD (per OD or per origin) ----------
if BLOCK == 'origin':
    # origin = OD index // (approx destinations); we lack an explicit O map,
    # so fall back to OD blocks (finest, strongest sparsity)
    blk = od_minor
else:
    blk = od_minor
# per-block rank so the TOTAL latent dim is comparable to a few per block
RB = 1 if n_od > 1000 else max(1, R // max(1, n_od))
t0 = time.time()
cols = []
order = np.argsort(blk, kind='stable')
uniq, starts = np.unique(blk[order], return_index=True)
starts = list(starts) + [n_minor]
Ublocks = []
rows_all = []
col_off = 0
data_list, row_list, col_list = [], [], []
for i, b in enumerate(uniq):
    idx = order[starts[i]:starts[i+1]]      # minor paths of this block
    if len(idx) == 0:
        continue
    sub = B2[idx]                           # nb x m
    kb = min(RB, min(sub.shape) - 1) if min(sub.shape) > 1 else 1
    if kb >= 1 and min(sub.shape) > 1:
        ub, sb, _ = svds(sub.astype(float), k=kb)
    else:
        # single row: unit vector
        ub = np.ones((len(idx), 1)) / np.sqrt(len(idx))
    for c in range(ub.shape[1]):
        for r_local, p in enumerate(idx):
            data_list.append(ub[r_local, c]); row_list.append(p); col_list.append(col_off)
        col_off += 1
tb = time.time() - t0
Ub = csr_matrix((data_list, (row_list, col_list)), shape=(n_minor, col_off))
Db = (B2.T @ Ub)                            # m x col_off, SPARSE
nnz_Db = Db.nnz if hasattr(Db, 'nnz') else int((np.abs(np.asarray(Db)) > 1e-9).sum())
print(f"{BLOCK.upper():6}-BLOCK SVD: {col_off:,} latent dims ({RB}/block)  "
      f"build {tb:6.2f}s  nnz(D)={nnz_Db:,} "
      f"({100*nnz_Db/(Db.shape[0]*Db.shape[1]):.3f}% dense)  "
      f"recon_err={recon_err(Ub):.4e}")

print(f"\nSVD build speedup (global/block): {tg/max(tb,1e-6):.1f}x")
print(f"decoder-D density ratio (global/block): "
      f"{(nnz_Dg/Dg.size)/(nnz_Db/(Db.shape[0]*Db.shape[1])+1e-30):.0f}x denser (global)")
