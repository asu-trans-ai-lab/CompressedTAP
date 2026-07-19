# Method note

For a column matrix \(B\), replace the original decision vector by

\[
x=Wz,
\]

where every column of \(W\) is a nonnegative convex combination of original columns. The effective matrix and cost are

\[
\widetilde B=BW,\qquad \widetilde c=W^Tc.
\]

The **RC** construction appends unit-vector columns for selected major variables and latent convex atoms for all remaining minor variables.

The initial evaluation should report column reduction, objective deviation, feasibility residual, and later runtime. It should not claim integer equivalence.
