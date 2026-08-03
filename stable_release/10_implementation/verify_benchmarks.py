"""Cross-verification of link volumes, link travel times, and Beckmann
objective across: tap-b (Algorithm B, full network, github.com/spartalab/tap-b),
the published TNTP best-known flows (github.com/bstabler/TransportationNetworks),
and our pool solutions (latgp / deep GP).

usage: python verify_benchmarks.py sketch|regional
"""
import csv
import os
import re
import sys

import numpy as np

NET = sys.argv[1]
CFG = {
    'sketch': dict(tntp='tap-b/net/ChicagoSketch_net.txt',
                   tapb='tap-b/cs_flows_out.txt',
                   pub='ChicagoSketch_flow.tntp',
                   ours='../updated_TAPLite/data/03_chicago_sketch/link.csv',
                   dumps='../updated_TAPLite/data/cpp_problem_E2_full_rich',
                   sols=[('latgp', '../updated_TAPLite/kernel/v_sk_latgp.f64'),
                         ('gp-deep', '../updated_TAPLite/kernel/v_sk_gpref.f64'),
                         ('gp', '../updated_TAPLite/kernel/v_sk_gp.f64')]),
    'regional': dict(tntp='tap-b/net/ChicagoRegional_net.txt',
                     tapb='tap-b/cr_flows_out.txt',
                     pub=None,
                     ours='../updated_TAPLite/data/04_chicago_regional/link.csv',
                     dumps='../updated_TAPLite/data/cpp_problem_E0_baseline_regional',
                     sols=[('latgp', '../updated_TAPLite/kernel/v_e0_latgp.f64'),
                           ('gp', '../updated_TAPLite/kernel/v_e0_gp.f64')]),
}[NET]

# ---- TNTP net: (from,to) -> (cap, len, fftt, B, power)
tn = {}
order = []
with open(CFG['tntp']) as f:
    body = False
    for ln in f:
        if '<END OF METADATA>' in ln:
            body = True
            continue
        if not body or ln.strip().startswith('~') or not ln.strip():
            continue
        parts = ln.replace(';', ' ').split()
        if len(parts) < 7:
            continue
        t, h = int(parts[0]), int(parts[1])
        tn[(t, h)] = tuple(float(x) for x in parts[2:7])  # cap,len,fftt,B,pow
        order.append((t, h))
print(f'[{NET}] TNTP links: {len(tn):,}')

# ---- our link order + bpr dumps
ours_th = []
with open(CFG['ours']) as f:
    for r in csv.DictReader(f):
        ours_th.append((int(r['from_node_id']), int(r['to_node_id'])))
dd = CFG['dumps']
t0 = np.fromfile(os.path.join(dd, 'bpr_t0.f64'))
al = np.fromfile(os.path.join(dd, 'bpr_alpha.f64'))
be = np.fromfile(os.path.join(dd, 'bpr_beta.f64'))
cap = np.fromfile(os.path.join(dd, 'bpr_cap.f64'))
m = len(ours_th)
assert m == t0.size, f'{m} vs {t0.size}'

# ---- BPR parameter identity check (by from,to)
n_match = 0
dcap = dal = dbe = 0.0
miss = 0
for i, th in enumerate(ours_th):
    if th not in tn:
        miss += 1
        continue
    c2, ln2, ff2, b2, p2 = tn[th]
    dcap = max(dcap, abs(cap[i] - c2) / max(c2, 1))
    dal = max(dal, abs(al[i] - b2))
    dbe = max(dbe, abs(be[i] - p2))
    n_match += 1
print(f'param check: matched {n_match:,}/{m:,} links (missing {miss}); '
      f'max |dcap|/cap {dcap:.2e}, max |dalpha| {dal:.2e}, '
      f'max |dbeta| {dbe:.2e}')

def times(v):
    return t0 * (1.0 + al * np.power(np.maximum(v, 0.0) / cap, be))

def beck(v):
    vp = np.maximum(v, 0.0)
    return float(np.sum(t0 * vp + t0 * al * cap / (be + 1.0)
                        * np.power(vp / cap, be + 1.0)))

# ---- tap-b flows -> our order
tb = {}
pat = re.compile(r'\((\d+),(\d+)\)\s+([-\d.eE]+)')
with open(CFG['tapb']) as f:
    for ln in f:
        mm = pat.match(ln.strip())
        if mm:
            tb[(int(mm.group(1)), int(mm.group(2)))] = float(mm.group(3))
v_tapb = np.array([tb.get(th, np.nan) for th in ours_th])
nn = np.isnan(v_tapb).sum()
print(f'tap-b flows: {len(tb):,} parsed, {nn} unmatched in our order')
v_tapb = np.nan_to_num(v_tapb)

flows = [('tap-b (full net, 1e-8)', v_tapb)]

# ---- published best-known flows (Sketch only)
if CFG['pub']:
    pub = {}
    with open(CFG['pub']) as f:
        next(f)
        for ln in f:
            p = ln.split()
            if len(p) >= 3:
                pub[(int(p[0]), int(p[1]))] = float(p[2])
    v_pub = np.array([pub.get(th, 0.0) for th in ours_th])
    flows.append(('published best-known', v_pub))

for nm, path in CFG['sols']:
    if os.path.exists(path):
        flows.append((f'{nm} (fixed pool)', np.fromfile(path)))

# ---- pairwise panel vs tap-b
ref = v_tapb
t_ref = times(ref)
print(f'\nBeckmann (our BPR params):')
for nm, v in flows:
    print(f'  {nm:26s} {beck(v):18,.2f}')
print(f'\npanel vs tap-b full-network solution:')
print(f"{'solution':26s} {'R_v^2':>10s} {'E_v':>10s} {'R_t^2':>10s} {'E_t':>10s}")
for nm, v in flows[1:]:
    tv = times(v)
    r2v = 1 - np.sum((v - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)
    ev = np.linalg.norm(v - ref) / np.linalg.norm(ref)
    r2t = 1 - np.sum((tv - t_ref) ** 2) / np.sum((t_ref - t_ref.mean()) ** 2)
    et = np.linalg.norm(tv - t_ref) / np.linalg.norm(t_ref)
    print(f'{nm:26s} {r2v:10.6f} {ev:10.2e} {r2t:10.6f} {et:10.2e}')
print('\nnote: pool solutions solve the FIXED-POOL UE; their gap to the '
      'full-network UE measures pool adequacy, not solver error. tap-b '
      'generalized cost includes toll/distance factors our BPR-only '
      'objective omits; Beckmann rows use OUR params on all flow vectors.')
