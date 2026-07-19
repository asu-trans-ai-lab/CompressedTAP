from __future__ import annotations
import numpy as np
from scipy.optimize import linprog


def solve_covering_lp(B, cost, demand, W=None):
    B=np.asarray(B,float); cost=np.asarray(cost,float); demand=np.asarray(demand,float)
    if W is None:
        Be, ce, R = B, cost, np.eye(B.shape[1])
    else:
        W=np.asarray(W,float); Be, ce, R = B@W, W.T@cost, W
    r=linprog(ce,A_ub=-Be,b_ub=-demand,bounds=[(0,None)]*Be.shape[1],method='highs')
    if not r.success: raise RuntimeError(r.message)
    x=R@r.x
    return {'objective':float(cost@x),'x':x,'z':r.x,'residual':B@x-demand}


def _dist(B,i,j):
    a=B[:,i].astype(float); b=B[:,j].astype(float)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 and nb==0:return 0.0
    if na==0 or nb==0:return 1.0
    return float(np.linalg.norm(a/na-b/nb))


def greedy_groups(B, ids, n_groups):
    ids=list(ids); n_groups=max(1,min(n_groups,len(ids)))
    seeds=[ids[0]]
    while len(seeds)<n_groups:
        rem=[j for j in ids if j not in seeds]
        seeds.append(max(rem,key=lambda j:min(_dist(B,j,s) for s in seeds)))
    groups=[[] for _ in seeds]
    for j in ids:
        k=min(range(len(seeds)),key=lambda g:_dist(B,j,seeds[g]))
        groups[k].append(j)
    return groups


def make_atom_matrix(n_columns, groups, costs):
    costs=np.asarray(costs,float); W=np.zeros((n_columns,len(groups)))
    for m,g in enumerate(groups):
        g=list(g); shifted=costs[g]-costs[g].min()+1.0
        w=1.0/shifted; w/=w.sum(); W[g,m]=w
    return W


def make_major_latent_matrix(B,costs,major_indices,n_minor_atoms):
    n=B.shape[1]; major=sorted(set(map(int,major_indices))); minor=[i for i in range(n) if i not in major]
    cols=[]
    for i in major:
        e=np.zeros(n); e[i]=1; cols.append(e)
    if minor:
        Wm=make_atom_matrix(n,greedy_groups(B,minor,n_minor_atoms),costs)
        cols.extend(Wm[:,j] for j in range(Wm.shape[1]))
    return np.column_stack(cols)


def relerr(v,r):
    return abs(v-r)/max(abs(r),1e-12)
