"""EPS figures for the ten-robot / ten-obstacle example."""
import sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from grl_core import Pot, circle_inside, DEG, wrap_deg

SEED, DEPTH = 3, 3
r = pickle.load(open(f'res_{SEED}_{DEPTH}.pkl', 'rb'))
b = pickle.load(open(f'br_{SEED}_{DEPTH}.pkl', 'rb'))
p, z, obs, N = r['p'], r['z'], r['obs'], 10
pot = Pot(r['Z'], r['P'], r['argK'], N)
inside = circle_inside(1.0, obs)
theta = b['theta']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'mathtext.fontset': 'cm',
                     'ps.fonttype': 42})
OBS_COL = plt.cm.tab10(np.arange(10))
OBS_FILL = [tuple(0.38 * np.array(c[:3]) + 0.62) for c in OBS_COL]   # opaque, pre-blended (EPS has no alpha)
GRID = dict(color='0.9', lw=0.5)

def plane(ax):
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12); ax.set_aspect('equal')
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    for k, ((c, rr)) in enumerate(obs):
        ax.add_patch(Circle(c, rr, fc=OBS_FILL[k], ec=OBS_COL[k], lw=0.9, zorder=3))
    ax.add_patch(Circle((0, 0), 1, fill=False, ec='k', lw=1.4, zorder=3))

def marks(ax, numbers=False):
    ax.plot(p.real, p.imag, 'kx', ms=6, mew=1.3, zorder=6)
    ax.plot(z.real, z.imag, 'ko', ms=6, mfc='w', mew=1.1, zorder=6)
    if numbers:
        for i in range(N):
            ax.annotate(str(i + 1), (p[i].real, p[i].imag), xytext=(-9 if p[i].real < 0 else 5, -3), textcoords='offset points', fontsize=7, ha='right' if p[i].real < 0 else 'left')

def ch(f, w, n=64):
    return np.concatenate([f(w[i:i + n]) for i in range(0, len(w), n)])

# ---------------- Fig. 1: field ----------------
n = 240
xs = np.linspace(-1, 1, n)
X, Y = np.meshgrid(xs, xs)
W = (X + 1j * Y).ravel()
mask = inside(W, 0.004)
dO = np.full(W.shape, np.nan + 0j); dO[mask] = ch(pot.dOm, W[mask])
grad = np.conj(dO).reshape(X.shape)                        # grad Phi = conj(Omega')
U, V = grad.real, grad.imag
sp = np.hypot(U, V); U, V = U / sp, V / sp
Phi = np.full(W.shape, np.nan)
def phi(w):
    w = w[:, None]
    return -np.log(np.abs(w - pot.Z)).sum(1) + np.log(np.abs(w - pot.P)).sum(1)
Phi[mask] = ch(phi, W[mask]); Phi = Phi.reshape(X.shape)

fig, ax = plt.subplots(figsize=(4.4, 4.0))
plane(ax)
ax.streamplot(X, Y, np.nan_to_num(U), np.nan_to_num(V), density=2.0, color='b', linewidth=0.55, arrowsize=0.6, zorder=2,
              start_points=None)
lv = np.nanpercentile(Phi, np.linspace(4, 96, 15))
ax.contour(X, Y, Phi, levels=lv, colors='r', linestyles='--', linewidths=0.5, zorder=2)
# mask streamlines inside obstacles / outside wall
for k, ((c, rr)) in enumerate(obs):
    ax.add_patch(Circle(c, rr, fc='w', ec='none', zorder=2.6))
    ax.add_patch(Circle(c, rr, fc=OBS_FILL[k], ec=OBS_COL[k], lw=0.9, zorder=3))
crit = r['crit']
ax.plot(crit.real, crit.imag, 'D', ms=3.2, mfc='0.25', mec='none', zorder=5)
marks(ax)
fig.tight_layout(pad=0.3)
fig.savefig('Forest_Omega_obs.eps'); fig.savefig('Forest_Omega_obs.png', dpi=200)

# ---------------- Fig. 2: branches and synchronous motion ----------------
# Positions at t = 0.2T, ..., 0.8T for k(t) = tan(pi t / 2T); R is scaled by a positive constant
# (which leaves GRL(theta) unchanged) so that k = 1 at the median mid-course gain.
mids = [np.median(br['Phi']) for br in b['br']]
c0 = np.median(mids)
fig, ax = plt.subplots(figsize=(4.4, 4.0))
plane(ax)
for i, br in enumerate(b['br']):
    w = br['pts']
    ax.plot(w.real, w.imag, color='b', lw=1.1, zorder=4)
levels = [-12.0, -4.0, 4.0, 12.0]          # common values of log k, relative to the mid-course gain
shades = ['0.75', '0.55', '0.35', '0.1']
for lv, sh in zip(levels, shades):
    # synchronous motion (item 4 of Theorem 1): all robots at the same gain k
    lk = c0 + lv
    for br in b['br']:
        w, F = br['pts'][1:-1], br['Phi']
        j = int(np.clip(np.searchsorted(F, lk), 1, len(F) - 1))
        a_ = (lk - F[j - 1]) / (F[j] - F[j - 1])
        q = w[j - 1] + a_ * (w[j] - w[j - 1])
        ax.plot(q.real, q.imag, 'o', ms=4.2, mfc=sh, mec='k', mew=0.4, zorder=5)
ts = [1, 2, 3, 4]
marks(ax, numbers=True)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', ms=4.2, mfc=sh, mec='k', mew=0.4, label=f'$k=k_{t}$') for t, sh in zip(ts, shades)]
ax.legend(handles=h, loc='lower center', ncol=4, fontsize=7, frameon=True, handletextpad=0.1, columnspacing=0.8,
          bbox_to_anchor=(0.5, -0.005), borderpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig('Forest_solution.eps'); fig.savefig('Forest_solution.png', dpi=200)

# check: monotonicity of Phi along branches (kappa diffeomorphism)
print('Phi monotone on all branches:', all(np.all(np.diff(br['Phi']) > 0) for br in b['br']))

# ---------------- Fig. 3: phases ----------------
grid, Tb = r['grid'], r['Theta_b']
def _arcd(th, lo, wd):
    if (th - lo) % 360 <= wd: return 0.0
    return min((lo - th) % 360, (th - lo - wd) % 360)
cm = np.array([min(_arcd(g, d['lo'], d['width']) for d in r['bnd']) for g in grid])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.4, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
for k, d in enumerate(r['bnd']):
    col = 'k' if k == 0 else OBS_COL[k - 1]
    lo, wd = d['lo'], d['width']
    for s in ([(lo, wd)] if lo + wd <= 180 else [(lo, 180 - lo), (-180, lo + wd - 360)]):
        a1.add_patch(plt.Rectangle((s[0], k - 0.35), s[1], 0.7, fc=(col if k == 0 else OBS_FILL[k - 1]), ec=(col if k == 0 else OBS_COL[k - 1]), lw=0.6))
for t in Tb:
    a1.plot([t, t], [-0.8, 10.8], color='r', lw=0.7, ls=':')
a1.axvline(theta, color='b', lw=1.2)
a2.text(theta + 4, 46, r'$\theta^\ast$', color='b', fontsize=10, va='center')
a1.set_yticks(range(11)); a1.set_yticklabels(['$C_0$'] + [f'$C_{{{k}}}$' for k in range(1, 11)], fontsize=6.5)
a1.set_ylim(10.8, -0.8); a1.grid(True, axis='x', **GRID)
a1.set_ylabel('boundary arcs', fontsize=8)
a2.plot(grid, np.clip(cm, -5, None), color='k', lw=0.8)
a2.fill_between(grid, 0, np.clip(cm, 0, None), color='0.85', lw=0)
a2.axhline(0, color='k', lw=0.5)
for t in Tb:
    a2.axvline(t, color='r', lw=0.7, ls=':')
a2.axvline(theta, color='b', lw=1.2)
a2.set_xlim(-180, 180); a2.set_xticks(range(-180, 181, 60))
a2.set_xlabel(r'$\theta$ [deg]'); a2.set_ylabel(r'$d(\theta,\Theta_\partial)$ [deg]', fontsize=8)
a2.grid(True, **GRID)
fig.tight_layout(pad=0.3, h_pad=0.4)
fig.savefig('Forest_phases.eps'); fig.savefig('Forest_phases.png', dpi=200)
print('ok')
