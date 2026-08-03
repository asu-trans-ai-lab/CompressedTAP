# Vendored upstream sources

Files here are **copies** of sources whose upstream lives outside any git repository, kept
in sync so the campaign runs from a clean clone. Since 2026-08-02 this directory IS the
import path: every committed driver inserts `vendor/` on `sys.path` (previously they pointed
at the untracked upstream location). If you change a file here, change its upstream too.

## `export_stage7.py`

Upstream: `source/updated_TAPLite/python/export_stage7.py` (untracked directory); vendored
byte-identical 2026-08-02. Exports a GMNS (dataset_dir, pool) problem to the C++ solver's
binary blobs; only dependency is `compressed_assignment.py` in this directory.

## `compressed_assignment.py`

Upstream: `source/updated_TAPLite/python/compressed_assignment.py` (untracked directory).

The only change made here is to the `build_compressed` docstring, and it is a correctness
matter rather than a cosmetic one. The previous text claimed the flow-weighted option
"minimizes the flow-weighted reconstruction loss". It does not: the implementation factorises
`S B2` with `S = diag(sqrt(f) + 1e-9)` and uses the left singular vectors **directly** as the
path-flow basis, whereas by Eckart-Young the minimiser of that loss is `span(S^-1 U~)`.
Measured at r=20 the achievable weighted loss is 249.1 as implemented, 209.1 for `S^-1 U~`,
and 227.9 for the plain SVD on Sioux Falls — so the implemented basis is worse at the stated
criterion than not weighting at all. See `campaign_weighted/verify_weighted_basis.py`.

The behaviour is unchanged and deliberately so: the heuristic improves the *solved* feasible
objective gap in 27 of 34 paired comparisons, which is the quantity that matters. Only the
claim was wrong, and the docstring now says what the code does.
