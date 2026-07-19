# Compression Benefits Prototype — Quick Summary

## E1. Route-rich break-even

At the largest tested route pool, `K=64`, the grouped
online kernel achieved a speedup of approximately
`1.70x` relative to the full reduced
kernel in this controlled NumPy benchmark.

Interpret this as a mechanism result: dimensional savings increasingly dominate
online data movement as route richness grows. Production speed claims should
continue to come from the unified C++/certified solver.

## E2. Memory and data movement

The suite reports online-storage and aggregate-link-signature size separately.
Grouped atoms can avoid retaining a full dense signed path decoder in the online
kernel; signed SVD retains that decoder when path nonnegativity must be checked.

## E3. Regularization

Across 20 mathematically
equivalent full-path solutions, the mean path-flow coefficient of variation was
`0.835`, while relative link-flow
variation was `1.824e-16` and the
relative objective range was
`6.223e-16`.

This demonstrates that path flows can be non-identifiable even when link flows
and the Beckmann objective are effectively unchanged. Grouped compression
selects one stable representative.

## E4. Adaptive promotion

Starting objective gap: `1.0764e-01%`.

Ending objective gap after promotion: `-6.3884e-14%`.

The nested promotion design preserves the previous feasible representation, so
the attainable objective should improve monotonically up to numerical tolerance.

## E5. Repeated scenarios

The basis/grouping is constructed once and reused across demand perturbations.
Use the cumulative-time plot to show preprocessing amortization, but use
production-engine timings for final speedup claims.

## Accuracy per variable

In the route-rich grid, weighted SVD matched or improved unweighted SVD in
`4/4` tested K configurations at the same
rank and under the same solver/constraints.

## Paper mapping

- **Computational reduction:** E1
- **Memory/data movement:** E2
- **Regularization/stability:** E3
- **Adaptive major-minor management:** E4
- **Repeated-scenario scalability:** E5
- **Accuracy per retained variable:** E1 objective-gap columns, supplemented by
  the existing Step 2A 32-case result
