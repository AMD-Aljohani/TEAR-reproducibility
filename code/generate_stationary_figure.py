from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

K=1.0
beta1=5.0
gamma=2.5
N=240
nodes, weights = leggauss(N)
x=(nodes+1)*K/2
w=weights*K/2
r=beta1-gamma*x
# Operator columns: f(y_i | x_j)*w_j
Y=x[:,None]
R=r[None,:]
A=(R*np.exp(-R*Y)/(1-np.exp(-R*K)))*w[None,:]
vals, vecs=np.linalg.eig(A)
j=int(np.argmax(vals.real))
pi=np.maximum(vecs[:,j].real,0)
if pi.sum()==0:
    pi=-vecs[:,j].real
pi=pi/np.sum(pi*w)
# Simulate retained trajectory
rng=np.random.default_rng(20260803)
burn=3000
n=600000
path=np.empty(n+burn+1)
path[0]=0.5
u=rng.random(n+burn)
for t in range(n+burn):
    rt=beta1-gamma*path[t]
    a=-math.expm1(-rt*K)
    path[t+1]=-math.log1p(-float(u[t])*a)/rt
ret=path[burn+1:]
fig,ax=plt.subplots(figsize=(6.2,4.2))
ax.hist(ret,bins=60,density=True,alpha=0.35,label='Empirical (600,000 retained steps)')
order=np.argsort(x)
ax.plot(x[order],pi[order],linewidth=1.8,label='Nyström stationary density')
ax.set_xlabel('x')
ax.set_ylabel(r'$\pi(x)$')
ax.set_title(r'Stationary density: $\beta_1=5$, $\gamma=2.5$, $K=1$')
ax.legend(frameon=False)
fig.tight_layout()
out=Path(__file__).with_name('Fig2_stationary_density.pdf')
fig.savefig(out,bbox_inches='tight')
# Numerical audit
hist,edges=np.histogram(ret,bins=100,range=(0,K),density=True)
centers=(edges[:-1]+edges[1:])/2
interp=np.interp(centers,x[order],pi[order])
l1=float(np.sum(np.abs(hist-interp)*np.diff(edges)))
wall=float(np.interp(K,x[order],pi[order]))
print({'eigenvalue':float(vals[j].real),'integral':float(np.sum(pi*w)),'l1_100bin':l1,'wall_density':wall})
