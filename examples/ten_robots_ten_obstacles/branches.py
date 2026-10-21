"""Stage 2: trace the N branches of GRL(theta*) and the synchronous motion; geometric checks."""
import sys, time, pickle
sys.path.insert(0, '.')
import numpy as np
from grl_core import *

SEED, DEPTH = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (3, 3)
res = pickle.load(open(f'res_{SEED}_{DEPTH}.pkl', 'rb'))
theta = float(sys.argv[3]) if len(sys.argv) > 3 else res['theta']
p, z, obs = res['p'], res['z'], res['obs']
N = len(p)
pot = Pot(res['Z'], res['P'], res['argK'], N)
inside = circle_inside(1.0, obs)
Z, P = pot.Z, pot.P

def Phi(w):  # -log|R| with the normalisation |K| = 1
    w = np.atleast_1d(np.asarray(w, complex))
    return np.array([-(np.log(np.abs(x - Z)).sum()) + np.log(np.abs(x - P)).sum() for x in w])

t0 = time.time()
br = []
for i in range(N):
    pts, flags, hit, end = trace_branch(pot, i, theta, inside, 1.0, max_steps=20000)
    br.append(dict(pts=pts, flags=flags, hit=hit, end=end))
    print(f'robot {i+1}: steps {len(pts)}, inside {bool(flags.all())}, hit {hit}, end zero {end}  [{time.time()-t0:.0f}s]', flush=True)

sigma = [b['end'] for b in br]
ok = all(b['flags'].all() and b['hit'] is None for b in br) and sorted(sigma) == list(range(N))
print('sigma:', [s + 1 if s is not None else None for s in sigma], 'certified run:', ok)

# geometric margins
def dist_obst(w):
    d = 1 - np.abs(w)
    for (c, r) in obs:
        d = np.minimum(d, np.abs(w - complex(*c)) - r)
    return d
clear_b = min(dist_obst(b['pts'][1:-1]).min() for b in br)
dmin = np.inf
for i in range(N):
    for j in range(i + 1, N):
        A, B = br[i]['pts'][::2], br[j]['pts'][::2]
        for k0 in range(0, len(A), 500):
            dmin = min(dmin, np.abs(A[k0:k0 + 500, None] - B[None, :]).min())
lengths = [np.abs(np.diff(b['pts'])).sum() for b in br]
L0 = np.abs(p - z[np.array(sigma)]).sum() if ok else np.nan
print(f'min distance between paths {dmin:.4f}, min clearance to obstacles/wall {clear_b:.4f}, total length {sum(lengths):.3f}, straight-line (same assignment) {L0:.3f}')

# kappa along each branch (for synchronous snapshots)
for b in br:
    w = b['pts'][1:-1]
    b['Phi'] = Phi(w)
pickle.dump(dict(theta=theta, br=br, sigma=sigma, ok=ok, dmin=dmin, clear=clear_b, lengths=lengths, L0=L0),
            open(f'br_{SEED}_{DEPTH}.pkl', 'wb'))
print(f'done [{time.time()-t0:.0f}s]')
