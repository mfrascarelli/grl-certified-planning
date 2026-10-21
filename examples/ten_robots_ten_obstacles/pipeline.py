"""Ten robots, ten circular obstacles in the unit disk: certified GRL planning (Algorithm 1)."""
import sys, time, pickle
sys.path.insert(0, '.')
import numpy as np
from grl_core import *

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rng = np.random.default_rng(SEED)
N = 10
a_s = np.deg2rad(np.linspace(125, 235, N)); a_g = np.deg2rad(np.linspace(-55, 55, N))
p = 0.84 * np.exp(1j * a_s)
z = 0.84 * np.exp(1j * a_g)
obs = []
while len(obs) < 10:
    c = complex(rng.uniform(-0.5, 0.5), rng.uniform(-0.62, 0.62))
    r = rng.uniform(0.06, 0.1)
    if abs(c) + r > 0.8: continue
    if any(abs(c - complex(*o[0])) < r + o[1] + 0.07 for o in obs): continue
    if np.min(np.abs(np.concatenate([p, z]) - c)) < r + 0.12: continue
    obs.append(((c.real, c.imag), r))

t0 = time.time()
pot, circles = build_obstacle_potential(1.0, obs, p, z, DEPTH, 'mt')
inside = circle_inside(1.0, obs)
sing = np.concatenate([pot.Z, pot.P])
print(f'singularities: {len(sing)} ({len(sing) - 2 * N} images)')

def ch(f, w, n=64):
    w = np.atleast_1d(np.asarray(w, complex))
    return np.concatenate([f(w[i:i + n]) for i in range(0, len(w), n)])

import os
cf = f'crit_{SEED}_{DEPTH}.npz'
if os.path.exists(cf):
    _d = np.load(cf); crit, napc = _d['crit'], list(_d['napc'])
else:
    # ---------- critical points: Newton from a grid, count by argument principle ----------
    G = 50
    xs = np.linspace(-1, 1, G)
    w = (xs[:, None] + 1j * xs[None, :]).ravel()
    w = w[inside(w, 0.005)]
    with np.errstate(all='ignore'):
        done = np.zeros(0, complex)
        for it in range(40):
            step = ch(pot.dOm, w) / ch(pot.d2Om, w)
            w = w - step
            keep = np.isfinite(w) & (np.abs(w) < 2)
            conv = keep & (np.abs(step) < 1e-12)
            done = np.concatenate([done, w[conv]]); w = w[keep & ~conv]
            if len(w) == 0: break
        w = np.concatenate([done, w])
        ok = inside(w, 1e-6) & (np.abs(ch(pot.dOm, w)) < 1e-7)
        ok &= np.min(np.abs(w[:, None] - sing[None, :]), axis=1) > 1e-5
    crit = []
    for x in w[ok]:
        if all(abs(x - y) > 1e-6 for y in crit): crit.append(x)
    crit = np.array(crit)
    Theta_b = pot.theta(crit) if len(crit) else np.array([])

    def ap_count(shift):
        t = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
        tot = 0.0
        for C in circles:
            r = C['r'] - shift if C['outer'] else C['r'] + shift
            ww = C['c'] + r * np.exp(1j * t)
            val = np.mean(ch(pot.d2Om, ww, 256) / ch(pot.dOm, ww, 256) * r * np.exp(1j * t))
            tot += (1 if C['outer'] else -1) * val.real
        return 2 * N + tot
    napc = [ap_count(s) for s in (0.004, 0.008)]
    print(f'critical points found: {len(crit)}; argument principle: {napc}  [{time.time()-t0:.1f}s]')

    np.savez(cf, crit=crit, napc=np.array(napc))
Theta_b = pot.theta(crit) if len(crit) else np.array([])
print(f'critical points: {len(crit)}; argument principle: {napc}')

# ---------- boundary phases and (second-order) certificate ----------
# For w on C_i within arc length s <= h/2 of the sample z_l:
#   |Psi(w) - Psi(z_l)| <= int_0^{h/2} (|Omega'(z_l)| + s K_i) ds = (h/2)|Omega'(z_l)| + h^2 K_i / 8,
# with K_i = sum_xi dist(xi, C_i)^{-2} >= sup_{C_i} |Omega''|.
bnd = []
for C in circles:
    dist = np.abs(np.abs(sing - C['c']) - C['r'])
    L, K = float(np.sum(1 / dist)), float(np.sum(1 / dist ** 2))
    h = min(2 * np.sqrt(8 * np.deg2rad(0.05) / K), 2 * np.pi * C['r'] / 720)   # h^2 K/8 <= 0.05 deg
    M = int(np.ceil(2 * np.pi * C['r'] / h)); h = 2 * np.pi * C['r'] / M
    t = np.linspace(0, 2 * np.pi, M, endpoint=False)
    ww = C['c'] + C['r'] * np.exp(1j * t)
    ar = ch(pot.argR, ww, 32)
    d1 = ch(pot.dOm, ww, 32)
    psi = -ar
    tol = h / 2 * np.abs(d1) + h * h * K / 8                                     # radians, per sample
    th = DEG * (np.pi - np.unwrap(ar))
    dn = np.abs(np.real(d1 * np.exp(1j * t)))
    bnd.append(dict(name=C['name'], c=C['c'], r=C['r'], L=L, K=K, M=M, h=h, psi=psi, tol=tol,
                    lo=float(wrap_deg(th.min())), width=float(th.max() - th.min()), eps=float(dn.max()),
                    bound=float(np.pi * C['r'] * dn.max() * DEG), tolmax=float(tol.max() * DEG),
                    Lh=float(L * h / 2 * DEG)))
    print(f"{C['name']}: r={C['r']:.3f} eps={dn.max():.3e} width={bnd[-1]['width']:.2f} bound={bnd[-1]['bound']:.2f} "
          f"M={M} max tol={tol.max()*DEG:.3f}deg (first-order L h/2 = {L*h/2*DEG:.1f}deg)  [{time.time()-t0:.0f}s]", flush=True)

grid = np.round(np.arange(-180, 180, 0.05), 3)
def cert_margin_all(grid):
    m = np.full(grid.shape, np.inf)
    g = np.deg2rad(grid) + np.pi
    for b in bnd:
        for i0 in range(0, len(b['psi']), 4000):
            ps, tl = b['psi'][i0:i0 + 4000], b['tol'][i0:i0 + 4000]
            d = np.abs(wrap_rad(ps[None, :] - g[:, None])) - tl[None, :]
            m = np.minimum(m, d.min(1) * DEG)
    return m
cm = cert_margin_all(grid)
db = np.array([np.abs(wrap_deg(g - Theta_b)).min() if len(Theta_b) else 180 for g in grid])
score = np.minimum(cm, db)
print(f'certified: {np.mean(cm > 0)*100:.1f}%  certified and not in Theta_b: {np.mean(score > 0)*100:.1f}%')
ib = int(np.argmax(score)); theta = float(grid[ib])
print(f'theta* = {theta:.2f} deg, certificate margin {cm[ib]:.2f} deg, distance from Theta_b {db[ib]:.2f} deg  [{time.time()-t0:.1f}s]')

pickle.dump(dict(p=p, z=z, obs=obs, depth=DEPTH, Z=pot.Z, P=pot.P, argK=pot.argK, crit=crit, Theta_b=Theta_b,
                 napc=napc, bnd=[{k: v for k, v in b.items() if k != 'psi'} for b in bnd],
                 
                 grid=grid, cm=cm, db=db, theta=theta), open(f'res_{SEED}_{DEPTH}.pkl', 'wb'))
