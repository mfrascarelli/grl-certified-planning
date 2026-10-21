import functools
import numpy as np

TAU = 2 * np.pi
DEG = 180 / np.pi

def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0

def wrap_rad(x):
    return (np.asarray(x) + np.pi) % TAU - np.pi

# ---------------- obstacle-free GRL: R = prod(z - zeta) / prod(z - p) ----------------
def crit_points(p, z):
    """Zeros of q = n'd - d'n (break-point candidates) and critical phases pi - arg R(b), in degrees."""
    n, d = np.poly(z), np.poly(p)
    q = np.polysub(np.polymul(np.polyder(n), d), np.polymul(np.polyder(d), n))
    q = np.trim_zeros(np.where(np.abs(q) < 1e-12 * np.abs(q).max(), 0, q), "f")
    if len(q) < 2:
        return np.array([], complex), np.array([])
    b = np.roots(q)
    R = np.polyval(n, b) / np.polyval(d, b)
    return b, wrap_deg(DEG * (np.pi - np.angle(R)))

def track(p, z, theta_deg, kmin=1e-4, kmax=1e4, steps=500, iters=8):
    """Continuation in k of the N roots of d(z) + k e^{j theta} n(z) = 0 (Durand-Kerner).
    Returns the paths (steps+1, N), the gains and the assignment robot -> goal."""
    p, z = np.asarray(p, complex), np.asarray(z, complex)
    n, d = np.poly(z), np.poly(p)
    e = np.exp(1j * theta_deg / DEG)
    ks = np.geomspace(kmin, kmax, steps + 1)
    roots = p.copy()
    out = np.empty((steps + 1, len(p)), complex)
    for s, k in enumerate(ks):
        c = d + k * e * n
        for _ in range(iters):
            diff = roots[:, None] - roots[None, :]
            np.fill_diagonal(diff, 1.0)
            roots = roots - np.polyval(c, roots) / c[0] / diff.prod(axis=1)
        out[s] = roots
    assign = np.array([int(np.argmin(np.abs(z - r))) for r in roots])
    return out, ks, assign

@functools.lru_cache(maxsize=64)
def family_cached(p_t, z_t):
    """GRL(theta) for theta = -175:10:175 (grey background family)."""
    return [track(list(p_t), list(z_t), t, 1e-3, 1e3, 160, 6)[0] for t in range(-175, 180, 10)]

def min_clearance(paths):
    N = paths.shape[1]
    best = (np.inf, None, None)
    sub = paths[:: max(1, len(paths) // 250)]
    for i in range(N):
        for j in range(i + 1, N):
            D = np.abs(sub[:, i][:, None] - sub[:, j][None, :])
            a, b = np.unravel_index(np.argmin(D), D.shape)
            if D[a, b] < best[0]:
                best = (float(D[a, b]), sub[a, i], sub[b, j])
    return best

@functools.lru_cache(maxsize=32)
def simulate_sync(p_t, z_t, theta_deg, T=1.0, nt=3000, t0=2e-4, frac=0.999):
    """Synchronous motion with the gain law k(t) = tan(pi t / 2T): RK4 integration of
    z' = kdot(t) / (kappa(z) Omega'(z)), i.e. lambda = kdot/(kappa |grad Phi|^2),
    started at the roots of 1 + k(t0) e^{j theta} R = 0 (t0 -> 0 is the start at the poles)."""
    p, z = np.asarray(p_t, complex), np.asarray(z_t, complex)
    n, d = np.poly(z), np.poly(p)
    e = np.exp(1j * theta_deg / DEG)

    def kgain(t):
        return np.tan(np.pi * t / (2 * T))

    def kdot(t):
        return np.pi / (2 * T) * (1 + kgain(t) ** 2)

    x0 = np.roots(d + kgain(t0) * e * n)
    paths, _, _ = track(p, z, theta_deg, 1e-6, kgain(t0), 200, 8)  # label the initial roots by robot
    x = np.array([x0[np.argmin(np.abs(x0 - r))] for r in paths[-1]])

    def kap(w):
        return np.abs(np.polyval(d, w) / np.polyval(n, w))

    def dOm(w):
        return (1 / (w[:, None] - p[None, :])).sum(1) - (1 / (w[:, None] - z[None, :])).sum(1)

    def f(t, w):
        return kdot(t) / (kap(w) * dOm(w))

    ts = np.linspace(t0, frac * T, nt + 1)
    dt = ts[1] - ts[0]
    X = np.empty((nt + 1, len(x)), complex)
    X[0] = x
    for s_ in range(nt):
        t = ts[s_]
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[s_ + 1] = x
    K = kap(X.ravel()).reshape(X.shape)
    ts = np.concatenate([[0.0], ts])
    X = np.vstack([p[None, :], X])
    K = np.vstack([np.zeros((1, len(p))), K])
    rho = np.array([np.prod(pi - z) / np.prod(np.delete(pi - p, i)) for i, pi in enumerate(p)])
    dR = np.array([np.prod(np.delete(zl - z, l)) / np.prod(zl - p) for l, zl in enumerate(z)])
    return ts, X, K, T, rho, dR

# ---------------- general rational R with extra (image) singularities ----------------
class Pot:
    """R(z) = e^{j argK} prod(z - Z) / prod(z - P); Omega = -log R."""

    def __init__(self, Z, P, argK=0.0, n_orig=None):
        self.Z, self.P, self.argK = np.asarray(Z, complex), np.asarray(P, complex), float(argK)
        self.n_orig = len(Z) if n_orig is None else n_orig

    def argR(self, w):
        w = np.asarray(w, complex)[..., None]
        return self.argK + np.angle(w - self.Z).sum(-1) - np.angle(w - self.P).sum(-1)

    def theta(self, w):
        """Phase of the unique GRL through w: theta(w) = pi - arg R(w)."""
        return wrap_deg(DEG * (np.pi - self.argR(w)))

    def dOm(self, w):
        w = np.asarray(w, complex)[..., None]
        return (1 / (w - self.P)).sum(-1) - (1 / (w - self.Z)).sum(-1)

    def d2Om(self, w):
        w = np.asarray(w, complex)[..., None]
        return -(1 / (w - self.P) ** 2).sum(-1) + (1 / (w - self.Z) ** 2).sum(-1)

def reflect(c, r, a):
    return c + r * r / np.conj(a - c)

def build_obstacle_potential(r0, obs, p, z, depth, mode):
    """mode='plain': R = prod(z-zeta)/prod(z-p).
    mode='mt': iterated Milne-Thomson reflections. At depth s, the images created on C_l at depth s-1
    are reflected in every C_i, i != l. Each reflection of a balanced set (Z_s, P_s) in a circle (c, r)
    multiplies R by conj(R_s(c)) * prod(z - rho(Z_s)) / prod(z - rho(P_s))."""
    circles = [dict(c=0j, r=float(r0), name="C0", outer=True)] + [
        dict(c=complex(*o[0]), r=float(o[1]), name=f"C{i + 1}", outer=False) for i, o in enumerate(obs)
    ]
    z0, p0 = np.asarray(z, complex), np.asarray(p, complex)
    Z, P, argK = [z0], [p0], 0.0
    if mode == "mt":
        frontier = [(z0, p0, -1)]
        for _ in range(depth):
            nxt = []
            for zs, ps, frm in frontier:
                for i, C in enumerate(circles):
                    if i == frm:
                        continue
                    rz, rp = reflect(C["c"], C["r"], zs), reflect(C["c"], C["r"], ps)
                    argK -= np.angle(C["c"] - zs).sum() - np.angle(C["c"] - ps).sum()
                    Z.append(rz)
                    P.append(rp)
                    nxt.append((rz, rp, i))
            frontier = nxt
    return Pot(np.concatenate(Z), np.concatenate(P), argK, len(z0)), circles

def circle_inside(r0, obs):
    cs = [(complex(*o[0]), o[1]) for o in obs]

    def inside(w, m=0.0):
        w = np.asarray(w, complex)
        ok = np.abs(w) < r0 - m
        for c, r in cs:
            ok &= np.abs(w - c) > r + m
        return ok

    return inside

def boundary_phases(pot, curves, n=1440):
    """For each closed boundary curve: theta(z) along it (continuous lift), the excluded
    phase arc [lo, lo+width], the residual eps = max |dPhi/dn| and the bound pi r eps (circles)."""
    res = []
    for C in curves:
        if "w" in C:  # generic closed curve given by samples
            w = C["w"]
            eps, bound = np.nan, np.nan
        else:
            t = np.linspace(0, 2 * np.pi, n + 1)
            w = C["c"] + C["r"] * np.exp(1j * t)
            dn = np.abs(np.real(pot.dOm(w) * np.exp(1j * t)))  # |dPhi/dn| = |Re(Omega' n)|
            eps = float(dn.max())
            bound = float(np.pi * C["r"] * eps * DEG)
        th = DEG * (np.pi - np.unwrap(pot.argR(w)))
        width = float(min(th.max() - th.min(), 360.0))
        res.append(dict(C=C, w=w, th=th, lo=float(wrap_deg(th.min())), width=width, eps=eps, bound=bound))
    return res

def in_arc(theta, b):
    return b["width"] >= 359.99 or (theta - b["lo"]) % 360 <= b["width"] + 1e-9

def arc_dist(theta, b):
    if in_arc(theta, b):
        return 0.0
    a = (b["lo"] - theta) % 360
    c = (theta - (b["lo"] + b["width"])) % 360
    return float(min(a, c))

def crit_in_domain(pot, inside, box, G=28):
    """Zeros of Omega' in D by Newton's method from a grid (as in Algorithm 1)."""
    xs = np.linspace(box[0], box[1], G)
    ys = np.linspace(box[2], box[3], G)
    w = (xs[:, None] + 1j * ys[None, :]).ravel()
    w = w[inside(w, 0.02)]
    with np.errstate(all="ignore"):
        for _ in range(50):
            w = w - pot.dOm(w) / pot.d2Om(w)
        ok = np.isfinite(w)
        w = w[ok]
        ok = inside(w, 1e-5) & (np.abs(pot.dOm(w)) < 1e-8)
        sing = np.concatenate([pot.Z, pot.P])
        ok &= np.min(np.abs(w[:, None] - sing[None, :]), axis=1) > 1e-4
    out = []
    for x in w[ok]:
        if all(abs(x - y) > 1e-6 for y in out):
            out.append(x)
    b = np.array(out, complex)
    return b, (pot.theta(b) if len(b) else np.array([]))

def trace_branch(pot, i, theta_deg, inside, scale, max_steps=5000):
    """Branch of GRL(theta) leaving the pole P[i]: gradient flow z' = conj(Omega')/|Omega'|
    (RK4 in arc length), projected after every step on the level set Psi = theta + pi.
    Returns points, inside-flags, first boundary hit (or None) and index of the end zero."""
    p = pot.P[i]
    th = theta_deg / DEG
    psi0 = float(wrap_rad(th + np.pi))
    others = np.delete(pot.P, i)
    ar = pot.argK + np.angle(p - pot.Z).sum() - np.angle(p - others).sum()
    dep = np.pi + th + ar  # departure direction, eq. (departure)
    h0, eps = scale / 140, 1e-3 * scale
    Z, P = pot.Z, pot.P

    def g(w):
        return (1 / (w - P)).sum() - (1 / (w - Z)).sum()

    def f(w):
        gw = g(w)
        return np.conj(gw) / abs(gw)

    x = p + eps * np.exp(1j * dep)
    pts, flags = [p], [True]
    hit, end, was_in, out_steps = None, None, True, 0
    for _ in range(max_steps):
        dZ = np.abs(x - Z)
        jz = int(dZ.argmin())
        dmin = min(dZ[jz], np.abs(x - P).min())
        if dZ[jz] <= dmin + 1e-15 and dZ[jz] < 0.5 * h0:
            end = jz
            pts.append(Z[jz])
            flags.append(bool(inside(Z[jz])))
            break
        h = max(2e-5 * scale, min(h0, 0.3 * dmin))
        k1 = f(x)
        k2 = f(x + h / 2 * k1)
        k3 = f(x + h / 2 * k2)
        k4 = f(x + h * k3)
        nx = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        e = float(wrap_rad(-pot.argR(nx) - psi0))
        gn = g(nx)
        q = abs(gn) ** 2
        if q > 1e-20 and abs(e) < 0.5:
            nx = nx - e * (gn.imag + 1j * gn.real) / q
        is_in = bool(inside(nx))
        if was_in and not is_in and hit is None:
            a, b = x, nx
            for _ in range(40):
                mid = (a + b) / 2
                if inside(mid):
                    a = mid
                else:
                    b = mid
            hit = a
            pts.append(a)
            flags.append(True)
        x, was_in = nx, is_in
        pts.append(x)
        flags.append(is_in)
        if not is_in:
            out_steps += 1
            if out_steps > 900:
                break
        if abs(x) > 4 * scale:
            break
    return np.array(pts), np.array(flags), hit, end

def argument_principle_count(pot, circles, n_orig, m=4096, shift=0.01):
    """#zeros of Omega' in D = #poles of Omega' in D + (1/2 pi j) oint_{dD} Omega''/Omega' dz.
    Outer circle counter-clockwise, obstacles clockwise; Omega' has the 2N original simple poles in D.
    The contour is moved by `shift` into D, so that zeros of Omega' on dD (stagnation points) are not on it."""
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    total = 0.0
    for C in circles:
        r = C["r"] - shift if C["outer"] else C["r"] + shift
        w = C["c"] + r * np.exp(1j * t)
        val = np.mean(pot.d2Om(w) / pot.dOm(w) * r * np.exp(1j * t))
        total += (1 if C["outer"] else -1) * val.real
    return 2 * n_orig + total

