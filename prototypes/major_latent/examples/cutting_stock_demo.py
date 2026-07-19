from __future__ import annotations
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from common import *


def enumerate_patterns(L,lengths):
    bounds=[L//a for a in lengths]; pats=[]
    for p in itertools.product(*(range(b+1) for b in bounds)):
        used=sum(a*n for a,n in zip(lengths,p))
        if 0<used<=L and all(used+a>L for a in lengths): pats.append(p)
    return np.asarray(pats,float).T


def main():
    L=110; lengths=[20,45,50,55]; demand=np.array([48,35,24,10.])
    B=enumerate_patterns(L,lengths); waste=L-np.asarray(lengths)@B; cost=np.ones(B.shape[1])+0.002*waste
    H0=solve_covering_lp(B,cost,demand)
    Wc=make_atom_matrix(B.shape[1],greedy_groups(B,range(B.shape[1]),5),cost)
    C=solve_covering_lp(B,cost,demand,Wc)
    score=cost-0.15*(B/demand[:,None]).sum(axis=0); major=np.argsort(score)[:4]
    Wrc=make_major_latent_matrix(B,cost,major,4); RC=solve_covering_lp(B,cost,demand,Wrc)
    print('\n=== Cutting stock ===')
    print(f'columns H0/C/RC: {B.shape[1]}/{Wc.shape[1]}/{Wrc.shape[1]}')
    print(f'objective H0/C/RC: {H0["objective"]:.4f}/{C["objective"]:.4f}/{RC["objective"]:.4f}')
    print(f'relative error C/RC: {relerr(C["objective"],H0["objective"]):.2%}/{relerr(RC["objective"],H0["objective"]):.2%}')
    print(f'min coverage residual H0/C/RC: {H0["residual"].min():.4g}/{C["residual"].min():.4g}/{RC["residual"].min():.4g}')

if __name__=='__main__': main()
