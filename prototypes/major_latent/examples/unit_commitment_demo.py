from __future__ import annotations
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize import linprog
from common import *


def valid(bits,min_up,min_down):
    s=0; n=len(bits)
    while s<n:
        e=s+1
        while e<n and bits[e]==bits[s]:e+=1
        run=e-s
        if s>0 and e<n:
            if bits[s] and run<min_up:return False
            if not bits[s] and run<min_down:return False
        s=e
    return any(bits)


def schedules(T,pmax,var,startup,min_up,min_down):
    cols=[]; costs=[]
    for bits in itertools.product([0,1],repeat=T):
        if not valid(bits,min_up,min_down):continue
        starts=sum(bits[t]==1 and (t==0 or bits[t-1]==0) for t in range(T))
        g=pmax*np.asarray(bits,float); cols.append(g); costs.append(var*g.sum()+startup*starts)
    return np.column_stack(cols),np.asarray(costs,float)


def solve(blocks,costs,demand,Ws=None):
    eB=[]; ec=[]; R=[]
    for i,(B,c) in enumerate(zip(blocks,costs)):
        W=None if Ws is None else Ws[i]
        eB.append(B if W is None else B@W); ec.append(c if W is None else W.T@c); R.append(np.eye(B.shape[1]) if W is None else W)
    n=sum(B.shape[1] for B in eB); Aeq=np.zeros((len(eB),n)); off=0
    for g,B in enumerate(eB):Aeq[g,off:off+B.shape[1]]=1; off+=B.shape[1]
    r=linprog(np.concatenate(ec),A_ub=-np.hstack(eB),b_ub=-demand,A_eq=Aeq,b_eq=np.ones(len(eB)),bounds=[(0,None)]*n,method='highs')
    if not r.success:raise RuntimeError(r.message)
    off=0; gen=np.zeros_like(demand); obj=0
    for B,c,W,Be in zip(blocks,costs,R,eB):
        z=r.x[off:off+Be.shape[1]]; x=W@z; gen+=B@x; obj+=c@x; off+=Be.shape[1]
    return {'objective':float(obj),'residual':gen-demand,'generation':gen}


def main():
    T=6; demand=np.array([85,120,145,135,105,75.])
    specs=[(90,18,80,2,1),(70,24,45,1,1),(50,36,20,1,1)]
    blocks=[]; costs=[]
    for p,v,s,u,d in specs:
        B,c=schedules(T,p,v,s,u,d); blocks.append(B); costs.append(c)
    H0=solve(blocks,costs,demand)
    Wc=[]; Wrc=[]
    for B,c in zip(blocks,costs):
        # Keep the maximum-capacity schedule as a singleton atom so the compressed master preserves adequacy.
        peak=int(np.argmax(B.sum(axis=0)))
        remaining=[i for i in range(B.shape[1]) if i!=peak]
        groups=[[peak]] + greedy_groups(B,remaining,min(5,len(remaining)))
        Wc.append(make_atom_matrix(B.shape[1],groups,c))
        score=c/np.maximum(B.sum(axis=0),1); major=np.argsort(score)[:min(3,B.shape[1])]
        Wrc.append(make_major_latent_matrix(B,c,major,min(5,max(1,B.shape[1]-len(major)))))
    C=solve(blocks,costs,demand,Wc); RC=solve(blocks,costs,demand,Wrc)
    print('\n=== Unit commitment schedule pool ===')
    print(f'columns H0/C/RC: {sum(B.shape[1] for B in blocks)}/{sum(W.shape[1] for W in Wc)}/{sum(W.shape[1] for W in Wrc)}')
    print(f'objective H0/C/RC: {H0["objective"]:.4f}/{C["objective"]:.4f}/{RC["objective"]:.4f}')
    print(f'relative error C/RC: {relerr(C["objective"],H0["objective"]):.2%}/{relerr(RC["objective"],H0["objective"]):.2%}')
    print(f'min demand residual H0/C/RC: {H0["residual"].min():.4g}/{C["residual"].min():.4g}/{RC["residual"].min():.4g}')

if __name__=='__main__':main()
