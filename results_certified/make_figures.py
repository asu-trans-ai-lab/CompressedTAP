#!/usr/bin/env python3
"""Reproduce the Section 6 figures from the certified CSVs in this directory.

    python make_figures.py            # renders whatever data is present

Figure A  speedup vs paths-per-OD (K/OD)   <- MASTER_DECOMPOSITION_SUMMARY.csv (Sioux rows)
Figure B  speedup vs target accuracy       <- e4w_curve_<tag>.csv (Chicago weighted sweep)

Both share the parity-line-at-1 design. Figure A is certified and always renders; Figure B
renders only once the accuracy-conditioned sweep has produced its curve CSV. Every number is
matched-accuracy by construction (see the CSV headers / README). Single hue per role, no
rainbow; the reference line at speedup 1 is the read: above it beats full-space ALM (H0).
"""
import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C_R, C_CR, C_RC = '#2a78d6', '#1baf7a', '#e87ba4'   # elimination / cond-compression / combined


def figure_a():
    rows = list(csv.DictReader(open(os.path.join(HERE, 'MASTER_DECOMPOSITION_SUMMARY.csv'))))
    sx = sorted((r for r in rows if r['instance'].startswith('sioux_SFK')
                 and r['basis'] == 'unweighted' and r['S_R']),
                key=lambda r: float(r['K_per_OD']))
    if not sx:
        print('Figure A: no Sioux rows'); return
    K = [float(r['K_per_OD']) for r in sx]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(1.0, ls='--', lw=1, color='#898781', zorder=1)
    for key, col, mk, lab in (('S_R', C_R, 'o', r'$S_R$  exact elimination'),
                              ('S_C|R', C_CR, '^', r'$S_{C|R}$  compression after elimination'),
                              ('S_RC', C_RC, 'D', r'$S_{RC}$  combined (R1 vs H0)')):
        y = [float(r[key]) for r in sx]
        ax.plot(K, y, marker=mk, color=col, lw=2, ms=6, mec='white', mew=1.2, label=lab,
                ls=('--' if key == 'S_C|R' else '-'))
    ax.set_xlabel('paths per OD  (K/OD)'); ax.set_ylabel('speedup vs full-space ALM (H0)')
    ax.set_ylim(0.4, 2.6); ax.grid(alpha=0.15); ax.legend(frameon=False, fontsize=9)
    ax.set_title('Sioux Falls: value decomposition vs richness', fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, 'figA_speedup_vs_kod.%s' % ext), dpi=150)
    print('Figure A -> figA_speedup_vs_kod.pdf/.png')


def figure_b():
    curves = sorted(glob.glob(os.path.join(HERE, 'e4w_curve_*.csv')))
    if not curves:
        print('Figure B: no e4w_curve_*.csv yet (accuracy sweep still running)'); return
    for path in curves:
        tag = os.path.basename(path)[len('e4w_curve_'):-4]
        rows = [r for r in csv.DictReader(open(path))]
        ranks = sorted({int(r['rank']) for r in rows})
        # headline rank = the one whose R1 reaches the tightest accuracy (smallest eps with a value)
        def tightest(rk):
            es = [float(r['eps']) for r in rows if int(r['rank']) == rk and r['S_C|R']]
            return min(es) if es else 1.0
        rk = min(ranks, key=tightest)
        sub = sorted((r for r in rows if int(r['rank']) == rk), key=lambda r: -float(r['eps']))
        eps = [float(r['eps']) * 100 for r in sub]   # as percent gap
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.axhline(1.0, ls='--', lw=1, color='#898781')
        for key, col, mk, lab in (('S_C', C_R, 'o', r'$S_C$  compression vs H0'),
                                  ('S_C|R', C_CR, '^', r'$S_{C|R}$  vs R0'),
                                  ('S_RC', C_RC, 'D', r'$S_{RC}$  combined vs H0')):
            xs = [e for e, r in zip(eps, sub) if r[key]]
            ys = [float(r[key]) for r in sub if r[key]]
            if xs:
                ax.plot(xs, ys, marker=mk, color=col, lw=2, ms=5, mec='white', mew=1,
                        label=lab, ls=('--' if key == 'S_C|R' else '-'))
        ax.set_xscale('log'); ax.invert_xaxis()   # loose -> tight (rightward)
        ax.set_xlabel('target accuracy  (reference objective gap, %)')
        ax.set_ylabel('speedup vs baseline'); ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title('Chicago %s: accuracy-conditioned speedup (weighted r=%d)' % (tag, rk),
                     fontsize=11)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(HERE, 'figB_speedup_vs_accuracy_%s.%s' % (tag, ext)),
                        dpi=150)
        print('Figure B -> figB_speedup_vs_accuracy_%s.pdf/.png (rank %d)' % (tag, rk))


if __name__ == '__main__':
    figure_a()
    figure_b()
