# Chicago Regional through the standard interface + C++ kernel (2026-07-18)
Protocol: identical to shipped crosscheck (n_od=3, seed 7, penalized-Dijkstra K paths,
v/c 1.05 calibration, nondimensionalized). Loader shims: case-tolerant vdf_fftt + make_factory
"chicago-regional" branch (recorded in package; no science change). tapkernel.exe: g++ -O2 -static.

| instance | columns | used arcs | parity (py vs cpp grouped obj) | C++ sparse kernel speedup |
|---|---|---|---|---|
| cr_k32 | 96 | 993  | rel 3.76e-14 | 1.675x |
| cr_k64 | 192 | 3,692 | rel 1.68e-08 | 2.621x |
All checks OK (objective, link-norm, reduced-gradient, grouped solve). REGIONAL CROSSCHECK PASSED.
Note: cr_k64 used-arc dimension (3,692) far exceeds cs_k64 (995) — the regional network enters
the large-incidence regime where the kernel/data-movement argument is strongest.
