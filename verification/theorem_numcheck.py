"""Quick numerical check of Theorem 1 on a random obstacle-free instance (N = 5).

Checks, in order:
  1. departure direction  arg(z - p_i) -> pi + theta + arg(rho_i)       (eq. departure)
  2. root continuation in k: each start ends at a distinct goal          (item 1, permutation)
  3. synchronous arrival: integrating z' = kappa/Omega' from the roots at k0,
     every gain follows k0/(1 - k0 t)                                     (item 4; explicit Euler, ~1% error)
  4. one Milne-Thomson reflection makes a circle a streamline (spread of arg R ~ 1e-15)

Run:  python theorem_numcheck.py      (requires numpy)
"""
import numpy as np
rng=np.random.default_rng(1); N=5
p=rng.uniform(-1,1,N)+1j*rng.uniform(-1,1,N); z=rng.uniform(-1,1,N)+1j*rng.uniform(-1,1,N)
th=0.37
n=np.poly(z); d=np.poly(p)
# departure angle check
R=lambda s: np.prod(s-z)/np.prod(s-p)
for i in range(N):
    rho=np.prod(p[i]-z)/np.prod(np.delete(p[i]-p,i))
    k=1e-6; r=np.roots(d+k*np.exp(1j*th)*n); r0=r[np.argmin(abs(r-p[i]))]
    print("dep err",np.angle(np.exp(1j*(np.angle(r0-p[i])-(np.pi+th+np.angle(rho))))))
# track roots, pairing
ks=np.logspace(-6,8,20000); cur=p.copy()
for k in ks:
    r=np.roots(d+k*np.exp(1j*th)*n)
    cur=np.array([r[np.argmin(abs(r-c))] for c in cur])
print("ends", [np.argmin(abs(z-c)) for c in cur])
# synchronous: dk/dt=k^2 integration of ODE with lambda=k/|grad|^2 for each robot
def grad(s): # conj(Omega'), Omega'=-R'/R = sum 1/(s-p)-sum 1/(s-z)
    return np.conj(np.sum(1/(s-p))-np.sum(1/(s-z)))
k0=0.5; r=np.roots(d+k0*np.exp(1j*th)*n); pos=r.copy(); T=1/k0; dt=1e-5; t=0
kap=lambda s: abs(1/R(s))
while t<0.999*T:
    for i in range(N):
        g=grad(pos[i]); pos[i]+=dt*kap(pos[i])*g/abs(g)**2
    t+=dt
print("gains at t", [kap(s) for s in pos], "expected", k0/(1-k0*t))
# single circle Milne-Thomson: Psi on circle constant
c=0.3+0.2j; rr=0.25
P=np.r_[p,c+rr**2/np.conj(p-c)]; Z=np.r_[z,c+rr**2/np.conj(z-c)]
# check none of originals inside circle
ok=all(abs(p-c)>rr) and all(abs(z-c)>rr)
w=c+rr*np.exp(1j*np.linspace(0,2*np.pi,400))
Rv=np.array([np.prod(s-Z)/np.prod(s-P) for s in w])
print("circle ok",ok,"spread of arg R on circle", np.ptp(np.unwrap(np.angle(Rv))))
