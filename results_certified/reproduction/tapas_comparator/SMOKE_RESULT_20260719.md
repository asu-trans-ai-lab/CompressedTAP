# TAsK_updated build + smoke certification (2026-07-19, Windows single-thread)

- Fresh build from source/comparators/TAsK_updated (upstream src + 5-line
  make_pair C++11 patch + boost_shim, g++ -O3 -static): BUILD OK.
- Smoke: Chicago Sketch TAPAS fast (precision 1e-6): 5 iterations,
  8.8 s, final relative gap 3.44e-07.
- Verification vs certified reference (cs_tapas_flows.txt, rel gap
  3.9e-8): 2,950 arcs, link rel-L2 3.553e-04, Beckmann objective
  agreement rel 1.985e-07.
- Chicago Regional TAPAS reference run launched detached (PID logged in
  task_regional_overnight_20260719.log; TIME_LIMIT 20000 s, precision
  1e-6) -- the missing reference for the corrected Regional column.
