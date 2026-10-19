import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Certified GRL planning: an interactive walkthrough",
)


@app.cell(hide_code=True)
def _():
    import functools

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 10,
            "axes.edgecolor": "#9aa6b4",
            "axes.labelcolor": "#16202c",
            "xtick.color": "#5a6879",
            "ytick.color": "#5a6879",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )
    return functools, matplotlib, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Certified GRL planning: an interactive walkthrough

    Companion notebook to the brief paper *Certified Coordinated Path Generation via Generalized Root Locus and Harmonic Potentials*
    (Automatica, revision R3). It shows the objects of the paper on small examples you can change, and it recomputes
    the key quantities with the same formulas used in the proofs.

    **How to use it.** Read top to bottom. Every interactive figure has its controls right above it. Nothing has to be
    installed: the notebook runs in the browser. All numerical routines are in the hidden cells; open them with the
    "show code" toggle of each cell.

    | Section | Paper | What you can see |
    |---|---|---|
    | 1. GRL as a level set | Sec. 2, Lemma 1 | one robot, the family of GRL arcs, the gain along a branch |
    | 2. Gradient motion | Sec. 2, Lemma 2 | what invariance gives, and what it does not |
    | 3. Critical phases $\Theta_b$ | Sec. 3 | break points, collisions and change of assignment |
    | 4. Main theorem | Sec. 3, Thm. 1 | statement and proof outline |
    | 5. Synchronous arrival | Thm. 1, item 4 | robots = closed-loop roots, finite arrival time |
    | 6. Boundary phases $\Theta_\partial$ | Sec. 3–4 | obstacles, one excluded phase interval per circle |
    | 7. Reflections and certificate | Sec. 4, Prop. 1 | residual, bound $\pi r\bar\varepsilon$, finite safety test |
    | 8. Reciprocal root locus | Sec. 4 | the classical RRL as a special case |
    | 9. Computational procedure | Sec. 4, Alg. 1 | critical points and their count by the argument principle |
    """)
    return


@app.cell(hide_code=True)
def _(functools, np):
    # =====================================================================
    # Numerical core (shared by all sections)
    # =====================================================================
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
    def simulate_sync(p_t, z_t, theta_deg, k0, nt=1500, frac=0.995):
        """RK4 integration of z' = kappa(z)/Omega'(z), i.e. lambda = kappa/|grad Phi|^2,
        from the N roots of 1 + k0 e^{j theta} R = 0."""
        p, z = np.asarray(p_t, complex), np.asarray(z_t, complex)
        n, d = np.poly(z), np.poly(p)
        e = np.exp(1j * theta_deg / DEG)
        x0 = np.roots(d + k0 * e * n)
        paths, _, _ = track(p, z, theta_deg, 1e-4, k0, 300, 8)  # label the initial roots by robot
        x = np.array([x0[np.argmin(np.abs(x0 - r))] for r in paths[-1]])

        def kap(w):
            return np.abs(np.polyval(d, w) / np.polyval(n, w))

        def dOm(w):
            return (1 / (w[:, None] - p[None, :])).sum(1) - (1 / (w[:, None] - z[None, :])).sum(1)

        def f(w):
            return kap(w) / dOm(w)

        T = 1 / k0
        ts = np.linspace(0, frac * T, nt + 1)
        dt = ts[1] - ts[0]
        X = np.empty((nt + 1, len(x)), complex)
        X[0] = x
        for s in range(nt):
            k1 = f(x)
            k2 = f(x + dt / 2 * k1)
            k3 = f(x + dt / 2 * k2)
            k4 = f(x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            X[s + 1] = x
        K = kap(X.ravel()).reshape(X.shape)
        return ts, X, K, T

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

    def argument_principle_count(pot, circles, n_orig, m=4096):
        """#zeros of Omega' in D = #poles of Omega' in D + (1/2 pi j) oint_{dD} Omega''/Omega' dz.
        Outer circle counter-clockwise, obstacles clockwise; Omega' has the 2N original simple poles in D."""
        t = np.linspace(0, 2 * np.pi, m, endpoint=False)
        total = 0.0
        for C in circles:
            w = C["c"] + C["r"] * np.exp(1j * t)
            val = np.mean(pot.d2Om(w) / pot.dOm(w) * C["r"] * np.exp(1j * t))
            total += (1 if C["outer"] else -1) * val.real
        return 2 * n_orig + total

    return (
        DEG,
        Pot,
        arc_dist,
        argument_principle_count,
        boundary_phases,
        build_obstacle_potential,
        circle_inside,
        crit_in_domain,
        crit_points,
        family_cached,
        in_arc,
        min_clearance,
        simulate_sync,
        trace_branch,
        track,
        wrap_deg,
    )


@app.cell(hide_code=True)
def _(DEG, matplotlib, np, plt):
    # =====================================================================
    # Plotting helpers
    # =====================================================================
    RC = ["#d2602a", "#2b8a5d", "#7a4fc4", "#1d82b5", "#b8437a", "#857512"]
    INK, MUTED, GRID, LINE = "#16202c", "#5a6879", "#e6ebf1", "#c4cfdb"
    DANGER, WARN, OK = "#c32f27", "#a86500", "#1f7a4d"
    CIRCLE_COL = ["#5a6879", "#b8437a", "#857512", "#1d82b5"]
    CYC = matplotlib.colormaps["twilight_shifted"]

    def phase_color(t):
        return CYC((np.asarray(t) + 180) / 360)

    def plane_axes(ax, lim, center=0j):
        ax.set_aspect("equal")
        ax.set_xlim(center.real - lim, center.real + lim)
        ax.set_ylim(center.imag - lim, center.imag + lim)
        ax.grid(True, color=GRID, lw=0.8)
        ax.axhline(0, color=LINE, lw=0.9, zorder=0)
        ax.axvline(0, color=LINE, lw=0.9, zorder=0)
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        ax.set_axisbelow(True)

    def draw_poles_zeros(ax, p, z, labels=True):
        for i, a in enumerate(p):
            ax.plot(a.real, a.imag, marker="x", ms=10, mew=2.6, color=RC[i % 6], zorder=6)
            if labels:
                ax.annotate(f"$p_{i + 1}$", (a.real, a.imag), xytext=(7, 5), textcoords="offset points", fontsize=10)
        for j, a in enumerate(z):
            ax.plot(a.real, a.imag, marker="o", ms=9, mfc="white", mec=INK, mew=1.8, zorder=6)
            if labels:
                ax.annotate(rf"$\zeta_{j + 1}$", (a.real, a.imag), xytext=(7, -13), textcoords="offset points", fontsize=10)

    def hue_background(ax, theta_fn, lim, center=0j, res=150, mask=None, alpha=0.32):
        xs = np.linspace(center.real - lim, center.real + lim, res)
        ys = np.linspace(center.imag - lim, center.imag + lim, res)
        W = xs[None, :] + 1j * ys[:, None]
        th = theta_fn(W)
        img = phase_color(th)
        img[..., 3] = alpha
        if mask is not None:
            img[~mask(W), 3] = alpha * 0.25
        ax.imshow(img, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", interpolation="bilinear", zorder=0)

    def phase_strip(ax, theta, bands=(), ticks=(), tick_color=DANGER, rows_labels=()):
        """Phase axis [-180, 180]: colour bar of theta, excluded bands (one row each), critical ticks, current theta."""
        ax.set_xlim(-180, 180)
        nb = len(bands)
        ax.set_ylim(-1.6 - nb, 1.3)
        grad = np.linspace(-180, 180, 361)
        ax.imshow(phase_color(grad)[None, :, :], extent=(-180, 180, 0.4, 1.1), aspect="auto", zorder=0)
        for r, (b, col) in enumerate(bands):
            y = -0.6 - r
            ax.broken_barh([(-180, 360)], (y - 0.3, 0.6), color=GRID)
            if b["width"] >= 359.99:
                ax.broken_barh([(-180, 360)], (y - 0.3, 0.6), color=col)
            else:
                lo, w = b["lo"], max(b["width"], 0.8)
                if lo + w > 180:
                    ax.broken_barh([(lo, 180 - lo), (-180, lo + w - 360 + 180)], (y - 0.3, 0.6), color=col)
                else:
                    ax.broken_barh([(lo, w)], (y - 0.3, 0.6), color=col)
            if r < len(rows_labels):
                ax.text(-184, y, rows_labels[r], ha="right", va="center", fontsize=9, color=col)
        for t in ticks:
            ax.plot([t, t], [-0.2 - nb - 0.25, -0.2 - nb + 0.45], color=tick_color, lw=2)
        ax.axvline(theta, color=INK, lw=1.6)
        ax.plot(theta, 1.2, marker="v", color=INK, ms=7, clip_on=False)
        ax.set_yticks([])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["−180°", "−90°", "0°", "+90°", "+180°"])
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)

    def fig_with_strip(size=(7.2, 8.2), ratio=(6.2, 1)):
        fig = plt.figure(figsize=size, layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=ratio)
        return fig, fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    def fmt_deg(t):
        return ("+" if t > 0 else "−" if t < 0 else "") + f"{abs(t):.1f}°"

    _ = DEG
    return (
        CIRCLE_COL,
        DANGER,
        INK,
        LINE,
        MUTED,
        OK,
        RC,
        WARN,
        draw_poles_zeros,
        fig_with_strip,
        fmt_deg,
        hue_background,
        phase_color,
        phase_strip,
        plane_axes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The GRL as a level set

    Starts $p_1,\dots,p_N$ and goals $\zeta_1,\dots,\zeta_N$ are the poles and zeros of

    $$R(z)=\frac{\prod_{i=1}^N (z-\zeta_i)}{\prod_{i=1}^N (z-p_i)},\qquad
    \Omega(z)=-\log R(z)=\Phi(z)+j\,\Psi(z).$$

    Let $\Sigma:=\{\zeta_1,\dots,\zeta_N,p_1,\dots,p_N\}$ be the set of zeros and poles of $R$. $\Omega$ is defined on
    $\mathbb{C}\setminus\Sigma$, and $\arg R$ is not defined at the points of $\Sigma$.

    $\Phi=-\log|R|$ is the scalar potential. $\Psi=-\arg R$ is the stream function. Both are harmonic, and by the
    Cauchy–Riemann relations $\nabla\Phi\perp\nabla\Psi$ and $\|\nabla\Phi\|=\|\nabla\Psi\|=|\Omega'|$.

    For a phase $\theta$, the Generalized Root Locus is

    $$\mathrm{GRL}(\theta)=\{z:\ 1+k\,e^{j\theta}R(z)=0 \text{ for some } k>0\}.$$

    **Reformulation used in all the proofs.** Put $F_\theta:=e^{j\theta}R$. The condition $1+kF_\theta=0$ reads
    $F_\theta(z)=-1/k$. As $k$ ranges over $(0,\infty)$, $-1/k$ ranges over the negative real axis, so

    $$\mathrm{GRL}(\theta)=\{z:\ F_\theta(z)<0\},\qquad \kappa(z):=-\frac{1}{F_\theta(z)}=e^{\Phi(z)} .$$

    A complex number is real negative exactly when its argument is $\pi$. The modulus is free and gives the gain.
    Hence (**Lemma 1**)

    $$z\in\mathrm{GRL}(\theta)\iff \arg R(z)\equiv\pi-\theta \iff \Psi(z)\equiv\theta+\pi .$$

    The map $z\mapsto\theta(z)=\pi-\arg R(z)$ has a single value modulo $2\pi$, so **through every point $z\notin\Sigma$
    passes exactly one GRL**. This simple fact is behind both excluded phase sets of Sections 3 and 6.

    **Example with one robot.** Take $p=-1$, $\zeta=1$. Then $\arg R(z)=\arg(z-1)-\arg(z+1)$ is the angle under
    which the segment $[-1,1]$ is seen from $z$. By the inscribed angle theorem, each $\mathrm{GRL}(\theta)$ is a
    circular arc from $p$ to $\zeta$:

    - $\theta=0$ gives the segment;
    - $\theta=\pm\pi/2$ gives the upper and lower unit semicircles;
    - $\theta\to\pm\pi$ gives wider and wider arcs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    theta1 = mo.ui.slider(-179, 179, step=1, value=60, label="phase θ (deg)", show_value=True, full_width=True)
    theta1
    return (theta1,)


@app.cell(hide_code=True)
def _(
    DEG,
    INK,
    LINE,
    RC,
    draw_poles_zeros,
    mo,
    np,
    plane_axes,
    plt,
    theta1,
    track,
):
    _p, _z = [-1 + 0j], [1 + 0j]
    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(9.4, 4.3), layout="constrained", width_ratios=(1.25, 1))
    plane_axes(_ax, 2.6)
    for _t in range(-170, 180, 10):
        _P, _, _ = track(_p, _z, _t, 1e-3, 1e3, 200, 6)
        _ax.plot(_P[:, 0].real, _P[:, 0].imag, color=LINE, lw=0.9, zorder=1)
    _P, _ks, _ = track(_p, _z, theta1.value, 1e-4, 1e4, 600, 8)
    _ax.plot(_P[:, 0].real, _P[:, 0].imag, color=RC[0], lw=2.6, zorder=3)
    _i = np.argmin(np.abs(np.log(_ks)))
    _ax.plot(_P[_i, 0].real, _P[_i, 0].imag, "o", color=RC[0], ms=7, zorder=5)
    _ax.annotate("κ = 1", (_P[_i, 0].real, _P[_i, 0].imag), xytext=(8, 6), textcoords="offset points", color=RC[0])
    draw_poles_zeros(_ax, np.array(_p), np.array(_z))
    _ax.set_title(f"GRL(θ) for θ = {theta1.value}°   (grey: θ every 10°)")
    _s = np.concatenate([[0], np.cumsum(np.abs(np.diff(_P[:, 0])))])
    _ax2.semilogy(_s / _s[-1], _ks, color=RC[0], lw=2.2)
    _ax2.set_xlabel("normalised arc length along the branch (p → ζ)")
    _ax2.set_ylabel("gain κ = e^Φ")
    _ax2.grid(True, which="both", color="#eef2f6")
    _ax2.set_title("κ grows monotonically from 0 (at p) to ∞ (at ζ)")
    _chk = np.angle((_P[_i, 0] - 1) / (_P[_i, 0] + 1)) * DEG
    _out = mo.vstack(
        [
            mo.as_html(_fig),
            mo.md(
                f"Check of Lemma 1 at the marked point: $\\arg R = {_chk:.2f}^\\circ$ and "
                f"$\\pi-\\theta = {((180 - theta1.value + 180) % 360) - 180:.2f}^\\circ$."
            ),
        ]
    )
    plt.close(_fig)
    _ = INK
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Gradient motion: what invariance gives and what it does not

    Each robot follows the gradient of the scalar potential,

    $$\dot z=\lambda(z)\,\nabla\Phi(z)^\top=-\lambda(z)\,\overline{R'(z)/R(z)},\qquad \lambda>0 \text{ locally Lipschitz}.$$

    **Lemma 2 (invariance).** Along every solution, $\dot\Psi=\lambda\,\nabla\Psi\cdot\nabla\Phi=0$ and
    $\dot\Phi=\lambda\|\nabla\Phi\|^2\ge 0$. A robot that starts on $\mathrm{GRL}(\theta)$ stays on it, and its gain
    $k(t)=e^{\Phi(z(t))}$ does not decrease.

    This is a **local** statement. By itself it leaves four questions open, and the main theorem answers all of them:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            1. **Where does the curve lead?** To a goal, around in a loop, or to the boundary?
            2. **How many curves are there, and who goes where?** Does every robot reach a *different* goal?
            3. **Can two curves meet?** If they can, two robots can collide.
            4. **Can a curve touch an obstacle?**
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Critical phases $\Theta_b$: when two branches meet

    A **critical point** is a point $b\notin\Sigma$ with $R'(b)=0$. Since $F_\theta'=e^{j\theta}R'$ and
    $e^{j\theta}\neq0$, the critical points of $F_\theta$ are the zeros of $R'$: they depend only on the starts and
    goals, not on $\theta$. What depends on $\theta$ is whether $\mathrm{GRL}(\theta)$ passes through them.

    - **Away from critical points.** Where $F_\theta'\neq0$, the holomorphic inverse function theorem makes
      $F_\theta$ a local biholomorphism. The preimage of the negative real axis is then locally one smooth arc.
    - **At a critical point.** $F_\theta(z)\approx F_\theta(b)+a(z-b)^2$. The preimage of the real axis is two
      perpendicular curves through $b$: this is the classical **break point**.

    Through each critical point passes exactly one GRL (Section 1), the one with phase $\pi-\arg R(b)$. Hence

    $$\Theta_b=\{\pi-\arg R(b)\ :\ R'(b)=0\}.$$

    **Finiteness.** $R'=(n'd-d'n)/d^2$, and $q=n'd-d'n$ has degree at most $2N-2$, because the leading terms
    $Nz^{2N-1}$ cancel. So there are at most $2N-2$ critical phases.

    **In the figure.** Each critical phase is a **red tick** on the phase strip. Choose "nearest critical phase" and
    watch two branches touch at the highlighted break point. The minimal distance between branches drops to zero,
    and the assignment becomes ambiguous. Then move the slider slightly to either side: the assignment
    $\sigma_\theta$ changes across the critical phase.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    preset3 = mo.ui.dropdown(
        options={
            "Two robots: crossing starts and goals": "two",
            "Four robots: two columns": "cols",
            "Four robots: scattered": "mix",
            "Random (four robots)": "rand",
        },
        value="Two robots: crossing starts and goals",
        label="configuration",
    )
    seed3 = mo.ui.slider(0, 50, value=7, label="random seed", show_value=True)
    theta3 = mo.ui.slider(-179, 179, step=0.5, value=12, label="phase θ (deg)", show_value=True, full_width=True)
    mode3 = mo.ui.radio(
        options=["use the slider", "nearest critical phase", "safest phase (max distance from Θb)"],
        value="use the slider",
        label="θ is set by",
        inline=True,
    )
    fam3 = mo.ui.checkbox(value=True, label="show the GRL family (one curve every 10°)")
    hue3 = mo.ui.checkbox(value=False, label="colour every point z by its phase θ(z)")
    mo.vstack([mo.hstack([preset3, seed3], justify="start", gap=2), theta3, mode3, mo.hstack([fam3, hue3], justify="start")])
    return fam3, hue3, mode3, preset3, seed3, theta3


@app.cell(hide_code=True)
def _(crit_points, mode3, np, preset3, seed3, theta3, wrap_deg):
    _PRE = {
        "two": ([-1, 1], [1j, -1j]),
        "cols": ([-1.6 + 1.3j, -1.5 + 0.35j, -1.7 - 0.5j, -1.5 - 1.4j], [1.5 + 1.4j, 1.7 + 0.4j, 1.5 - 0.45j, 1.6 - 1.3j]),
        "mix": ([-1.5 + 1j, -1.4 - 0.8j, 0.2 + 1.6j, 0.5 - 1.5j], [1.4 + 0.6j, 1.5 - 0.9j, -0.3 + 0.2j, 1.6 + 1.7j]),
    }
    if preset3.value == "rand":
        _rng = np.random.default_rng(seed3.value)
        _pts = []
        while len(_pts) < 8:
            _c = complex(*_rng.uniform(-1.8, 1.8, 2))
            if all(abs(_c - _q) > 0.55 for _q in _pts):
                _pts.append(_c)
        p3, z3 = np.array(_pts[:4]), np.array(_pts[4:])
    else:
        p3, z3 = (np.asarray(_v, complex) for _v in _PRE[preset3.value])
    crit3, thb3 = crit_points(p3, z3)
    if mode3.value == "nearest critical phase" and len(thb3):
        th3 = float(thb3[np.argmin(np.abs(wrap_deg(thb3 - theta3.value)))])
    elif mode3.value.startswith("safest"):
        _grid = np.arange(-150, 150.01, 0.25)
        _m = np.min(np.abs(wrap_deg(_grid[:, None] - thb3[None, :])), axis=1) if len(thb3) else 180 - np.abs(_grid)
        th3 = float(_grid[np.argmax(_m)])
    else:
        th3 = float(theta3.value)
    return crit3, p3, th3, thb3, z3


@app.cell(hide_code=True)
def _(
    DANGER,
    INK,
    LINE,
    MUTED,
    OK,
    RC,
    WARN,
    crit3,
    draw_poles_zeros,
    fam3,
    family_cached,
    fig_with_strip,
    fmt_deg,
    hue3,
    hue_background,
    min_clearance,
    mo,
    np,
    p3,
    phase_strip,
    plane_axes,
    plt,
    th3,
    thb3,
    track,
    wrap_deg,
    z3,
):
    _paths, _ks, assign3 = track(p3, z3, th3)
    _clear = min_clearance(_paths)
    _allp = np.concatenate([p3, z3])
    _ctr = complex(np.mean([_allp.real.min(), _allp.real.max()]), np.mean([_allp.imag.min(), _allp.imag.max()]))
    _lim = max(1.5, max(np.ptp(_allp.real), np.ptp(_allp.imag)) / 2 * 1.6 + 0.4)
    _fig, _ax, _axs = fig_with_strip()
    plane_axes(_ax, _lim, _ctr)
    _n, _d = np.poly(z3), np.poly(p3)
    if hue3.value:
        hue_background(_ax, lambda W: wrap_deg(180 - np.degrees(np.angle(np.polyval(_n, W) / np.polyval(_d, W)))), _lim, _ctr)
    if fam3.value:
        for _F in family_cached(tuple(p3), tuple(z3)):
            _ax.plot(_F.real, _F.imag, color=LINE, lw=0.8, zorder=1)
    for _i in range(len(p3)):
        _pp = np.concatenate([[p3[_i]], _paths[:, _i], [z3[assign3[_i]]]])
        _ax.plot(_pp.real, _pp.imag, color=RC[_i % 6], lw=2.6, zorder=3)
    if _clear[1] is not None:
        _ax.plot([_clear[1].real, _clear[2].real], [_clear[1].imag, _clear[2].imag], ls="--", color=INK, lw=1.1, zorder=4)
    _margin = float(np.min(np.abs(wrap_deg(thb3 - th3)))) if len(thb3) else np.inf
    for _b, _t in zip(crit3, thb3):
        _near = abs(wrap_deg(_t - th3)) < 3
        _ax.plot(_b.real, _b.imag, "D", ms=8 if _near else 5.5, color=DANGER if _near else MUTED, zorder=5)
        if _near:
            _ax.annotate("break point", (_b.real, _b.imag), xytext=(10, 8), textcoords="offset points", color=DANGER, weight="bold")
    draw_poles_zeros(_ax, p3, z3)
    _ax.set_title(f"GRL(θ), θ = {fmt_deg(th3)}")
    phase_strip(_axs, th3, ticks=thb3)
    _axs.set_title("phase axis: red ticks = critical phases Θb", fontsize=9.5, loc="left", color=MUTED, weight="normal")
    if _margin < 0.3:
        _status = mo.callout(mo.md("**θ ∈ Θb**: a break point lies on GRL(θ) and two robots meet there."), kind="danger")
    elif _margin < 5:
        _status = mo.callout(mo.md(f"Admissible, but only **{_margin:.2f}°** from a critical phase."), kind="warn")
    else:
        _status = mo.callout(mo.md(f"**Admissible phase**: distance from Θb = {_margin:.1f}°."), kind="success")
    _cnt = np.bincount(assign3, minlength=len(z3))
    _rows = "\n".join(
        f"| robot {i + 1} | $p_{i + 1}\\to\\zeta_{a + 1}$ | {'**contested**' if _cnt[a] > 1 else ''} |" for i, a in enumerate(assign3)
    )
    _tbl = mo.md(
        "| | assignment $\\sigma_\\theta$ | |\n|---|---|---|\n"
        + _rows
        + f"\n\n**Critical points:** {len(crit3)} (bound $2N-2={2 * len(p3) - 2}$).  "
        + "**Critical phases:** "
        + (", ".join(fmt_deg(t) for t in np.sort(thb3)) or "none")
        + f".  \n**Minimal distance between branches** (dashed segment): {_clear[0]:.3f}."
    )
    _out = mo.vstack([_status, mo.as_html(_fig), _tbl])
    plt.close(_fig)
    _ = (OK, WARN)
    _out
    return (assign3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The two-robot example by hand.** With $p=\pm1$ and $\zeta=\pm j$, $R=(z^2+1)/(z^2-1)$ and
    $R'(z)=-4z/(z^2-1)^2$. The only critical point is $b=0$. Since $R(0)=-1$, $\arg R(0)=\pi$ and $\Theta_b=\{0\}$:
    the classical root locus is exactly the bad phase.

    For $\theta=0$ the closed-loop equation gives $z^2=(1-k)/(1+k)$. Both robots reach the origin at $k=1$ and then
    leave along the imaginary axis. For $\theta\neq0$, $z^2=(1-w)/(1+w)$ with $w=ke^{j\theta}$ never vanishes, so the
    collision disappears. The sign of $\theta$ decides who goes where.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Main theorem: certified coordinated GRL motion

    **Notation.** For a domain $D$ (an open connected set), $\partial D$ is its boundary and
    $\overline D=D\cup\partial D$ is its closure. Applied to a set, the bar always denotes the closure (e.g.
    $\overline{B_i}$ below); applied to a number or a function, as in Sections 2 and 7, it denotes complex conjugation.
    A Jordan curve is a closed curve without self-intersections. A function is meromorphic if it is holomorphic except
    at isolated poles.

    **Assumption.** $D\subset\mathbb{C}$ is a bounded domain whose boundary is a finite union of disjoint
    piecewise-smooth Jordan curves. $R$ is meromorphic near $\overline D$, and its zeros and poles in $\overline D$
    are exactly $\zeta_1,\dots,\zeta_N$ and $p_1,\dots,p_N$, all simple and inside $D$. Further singularities
    **outside** $\overline D$ are allowed: they are the images of Section 7.

    **Excluded phases.** $\Theta_b$ (Section 3) and the **boundary phases**
    $\Theta_\partial=\{\Psi(z)-\pi:\ z\in\partial D\}$, which are the families that touch $\partial D$ (Section 6).
    A phase outside $\Theta_b\cup\Theta_\partial$ is **admissible**.

    **Theorem.** For every admissible $\theta$:

    1. **Branch structure.** $\mathrm{GRL}(\theta)\cap D$ consists of exactly $N$ disjoint smooth arcs $B_i$.
       Each $B_i$ joins $p_i$ to $\zeta_{\sigma(i)}$, where $\sigma=\sigma_\theta$ is a permutation. $\kappa$ maps
       $B_i$ diffeomorphically onto $(0,\infty)$, and $\overline{B_i}\cap\partial D=\emptyset$.
    2. **Convergence.** For any locally Lipschitz $\lambda_i>0$, $z_i(t)$ stays in $B_i$, $\kappa$ increases without
       bound, and $z_i\to\zeta_{\sigma(i)}$.
    3. **Safety.** $z_i(t)\neq z_j(\tau)$ for all $t,\tau$, and $z_i(t)\notin\partial D$.
    4. **Synchronous arrival.** With $\lambda_i=\kappa\|\nabla\Phi\|^{-2}$, $\kappa(t)=\kappa_0/(1-\kappa_0t)$ and the
       goal is reached at $T=1/\kappa_0$ with bounded speed. Robots started at the roots of
       $1+k_0e^{j\theta}R=0$ remain at the roots of $1+k(t)e^{j\theta}R=0$ (Section 5).

    Open the steps below for the idea of each part of the proof.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Step 1. Local structure (no crossings, no loops)": mo.md(
                r"""
                Let $z^\ast\in\mathrm{GRL}(\theta)\cap D$. Since $\theta\notin\Theta_b$, $F_\theta'(z^\ast)\neq0$. By the
                holomorphic inverse function theorem, $F_\theta$ is a bijection of a neighbourhood $U$ of $z^\ast$ onto a
                neighbourhood $V$ of $F_\theta(z^\ast)<0$, with holomorphic inverse $G$. So
                $\mathrm{GRL}\cap U=G(V\cap\mathbb{R}_{<0})$, which is one smooth arc along which $\kappa=-1/F_\theta$ is
                strictly monotone.

                Consequences:

                - The GRL in $D$ is an embedded one-dimensional manifold, and distinct components never meet.
                - A component cannot be a closed loop: on a loop, $\kappa$ would attain a maximum, which is a critical
                  point. So every component is an open arc with $\kappa(B)=(a,b)$.
                """
            ),
            "Step 2. Where a branch ends": mo.md(
                r"""
                Take $z_n\in B$ with $\kappa(z_n)\to b$, and a limit point $z^\ast\in\overline D$ (compactness). There
                are three cases:

                - $z^\ast\in\partial D$: then $F_\theta(z^\ast)<0$, so $\theta\in\Theta_\partial$. Excluded.
                - $z^\ast\in D\setminus\Sigma$: then $z^\ast$ is on the GRL, and by Step 1 the arc through it is $B$ itself.
                  This would give $b=\kappa(z^\ast)<b$. Contradiction.
                - $z^\ast$ is a pole: then $\kappa\to0\neq b$.

                Hence $b=\infty$ and $z^\ast$ is a zero. The limit set is connected and contained in a finite set, so it
                is a single zero. In the same way $a=0$ and the other end is a single pole.
                """
            ),
            "Step 3. Exactly N branches and a permutation": mo.md(
                r"""
                Near a simple pole $p_i$, $w=1/F_\theta$ is holomorphic with $w(p_i)=0$ and $w'(p_i)\neq0$. The local
                inverse then shows that exactly **one** arc of the GRL leaves $p_i$. In the same way, exactly one arc
                reaches each simple zero. Every component has one pole end and one zero end, so there are $N$ components,
                and $\sigma$ is a bijection.
                """
            ),
            "Step 4. Dynamics": mo.md(
                r"""
                By Lemma 2 the robot stays on its branch $B_i\cong(0,\infty)$, where $\kappa$ is used as the coordinate.
                The motion reduces to $\dot\kappa=g_i(\kappa)>0$. If $\kappa$ stayed bounded, the robot would remain in a
                compact subarc where $g_i$ is bounded away from zero, which is impossible. So $\kappa\to\infty$ and
                $z_i\to\zeta_{\sigma(i)}$.

                With $\lambda=\kappa/\|\nabla\Phi\|^2$, $\dot\kappa=\kappa^2$ (a Riccati equation), whose solution is
                $\kappa_0/(1-\kappa_0t)$. Near the goal $\kappa\simeq1/(|R'(\zeta)||z-\zeta|)$ and
                $\|\nabla\Phi\|\simeq1/|z-\zeta|$, so the speed tends to $1/|R'(\zeta)|$.
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Synchronous arrival: the robots are the closed-loop roots

    With the time law $\lambda_i=\kappa\,\|\nabla\Phi\|^{-2}$, the velocity is $\dot z=\kappa(z)/\Omega'(z)$ and
    $\dot\kappa=\kappa^2$. Below, the motion is **integrated numerically** (RK4) from the $N$ roots of
    $1+k_0e^{j\theta}R=0$, using the configuration and phase of Section 3. It is compared with two predictions:

    - **filled dots:** integrated robot positions at time $t$;
    - **hollow circles:** roots of $1+k(t)\,e^{j\theta}R(z)=0$ with $k(t)=k_0/(1-k_0t)$, computed independently.

    The two coincide, and every robot's gain follows the same curve, so all robots arrive together at $T=1/k_0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    k0_5 = mo.ui.slider(0.02, 0.5, step=0.01, value=0.05, label="initial gain k₀", show_value=True)
    tt5 = mo.ui.slider(0, 0.995, step=0.005, value=0.6, label="time t / T", show_value=True, full_width=True)
    mo.vstack([k0_5, tt5])
    return k0_5, tt5


@app.cell(hide_code=True)
def _(
    INK,
    RC,
    assign3,
    draw_poles_zeros,
    k0_5,
    mo,
    np,
    p3,
    plane_axes,
    plt,
    simulate_sync,
    th3,
    track,
    tt5,
    z3,
):
    _ts, _X, _K, _T = simulate_sync(tuple(p3), tuple(z3), float(th3), float(k0_5.value))
    _idx = int(round(tt5.value / 0.995 * (len(_ts) - 1)))
    _t = _ts[_idx]
    _kt = k0_5.value / (1 - k0_5.value * _t)
    _roots = np.roots(np.poly(p3) + _kt * np.exp(1j * np.radians(th3)) * np.poly(z3))
    _paths, _, _ = track(p3, z3, th3)
    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained", width_ratios=(1.15, 1))
    _allp = np.concatenate([p3, z3])
    _ctr = complex(np.mean([_allp.real.min(), _allp.real.max()]), np.mean([_allp.imag.min(), _allp.imag.max()]))
    plane_axes(_ax, max(1.5, max(np.ptp(_allp.real), np.ptp(_allp.imag)) / 2 * 1.6 + 0.4), _ctr)
    for _i in range(len(p3)):
        _pp = np.concatenate([[p3[_i]], _paths[:, _i], [z3[assign3[_i]]]])
        _ax.plot(_pp.real, _pp.imag, color=RC[_i % 6], lw=1.6, alpha=0.55, zorder=2)
        _ax.plot(_X[: _idx + 1, _i].real, _X[: _idx + 1, _i].imag, color=RC[_i % 6], lw=2.8, zorder=3)
        _ax.plot(_X[_idx, _i].real, _X[_idx, _i].imag, "o", ms=9, color=RC[_i % 6], zorder=6)
    _ax.plot(_roots.real, _roots.imag, "o", ms=15, mfc="none", mec=INK, mew=1.4, zorder=7)
    draw_poles_zeros(_ax, p3, z3)
    _ax.set_title(f"t = {_t:.2f}  (T = {_T:.1f}),  k(t) = {_kt:.3g}")
    for _i in range(len(p3)):
        _ax2.semilogy(_ts / _T, _K[:, _i], color=RC[_i % 6], lw=2.4, alpha=0.8)
    _ax2.semilogy(_ts / _T, k0_5.value / (1 - k0_5.value * _ts), ls="--", color=INK, lw=1.3, label="k₀ / (1 − k₀ t)")
    _ax2.axvline(_t / _T, color="#9aa6b4", lw=1)
    _ax2.set_xlabel("t / T")
    _ax2.set_ylabel("gain κ(z_i(t))")
    _ax2.legend(loc="upper left", frameon=False)
    _ax2.grid(True, which="both", color="#eef2f6")
    _ax2.set_title("all gains follow the same Riccati solution")
    _rel = np.max(np.abs(_K - (k0_5.value / (1 - k0_5.value * _ts))[:, None]) / (k0_5.value / (1 - k0_5.value * _ts))[:, None])
    _pos = max(np.min(np.abs(_roots - _x)) for _x in _X[_idx])
    _speed = np.abs(np.diff(_X, axis=0)).max() / (_ts[1] - _ts[0])
    _out = mo.vstack(
        [
            mo.as_html(_fig),
            mo.md(
                f"Maximum relative deviation of the gains from $k_0/(1-k_0t)$ over the whole run: **{_rel:.1e}**.  \n"
                f"Distance between integrated robots and closed-loop roots at the current time: **{_pos:.1e}**.  \n"
                f"Maximum speed along the run: **{_speed:.3f}** (bounded, as stated in item 4)."
            ),
        ]
    )
    plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Boundary phases $\Theta_\partial$: obstacles

    Following the RRL construction, the polygonal workspace is mapped conformally onto a **circular domain**
    $D=\{|z|<r_0\}\setminus\bigcup_i\{|z-c_i|\le r_i\}$, and all the analysis is done in $D$.

    **Definition.** A point $z\in\partial D$ lies on exactly one GRL, the one of phase $\theta(z)=\pi-\arg R(z)$. The
    boundary phases are the phases of all the GRLs that touch the boundary:

    $$\Theta_\partial=\{\theta(z):\ z\in\partial D\}.$$

    **One interval per circle.** An obstacle contains no pole and no zero, and the outer circle contains as many
    zeros as poles. So the winding of $R$ along every circle $C_i$ is zero, and $\theta(z)$ has a continuous
    single-valued lift on $C_i$. The image of a circle under a continuous function is a closed interval. Each circle
    therefore excludes one arc $[\theta_{\min},\theta_{\max}]$ of phases. The interval belongs to the circle, not to
    a robot: $R$ contains all robots at once.

    **What the figure shows.**

    - The **colour of every boundary circle** is $\theta(z)$ on that circle.
    - Each circle has its own **row of the phase strip** showing its excluded interval. **Ochre ticks** are the
      critical phases $\Theta_b$ in $D$.
    - When $\theta$ falls inside an interval, **red dots** mark where $\mathrm{GRL}(\theta)$ touches the boundary,
      and branch portions outside $D$ are dashed.

    With the plain potential $\Omega=-\log R$ the colour changes along every circle, and each circle excludes a
    large interval. In the disk scenario with one robot, $\Theta_\partial=\{|\theta|\ge126.87^\circ\}$: the wide arcs
    of Section 1 leave the disk. Switch to **"Milne–Thomson reflections"** to see the intervals shrink (Section 7).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    scen6 = mo.ui.dropdown(
        options={"One robot in a disk": "disk", "Two robots, one obstacle": "one", "Four robots, two obstacles": "two"},
        value="Two robots, one obstacle",
        label="scenario",
    )
    pot6 = mo.ui.radio(
        options={"plain Ω = −log R": "plain", "Milne–Thomson reflections": "mt"},
        value="plain Ω = −log R",
        label="potential",
        inline=True,
    )
    depth6 = mo.ui.slider(1, 3, value=2, label="reflection depth m", show_value=True)
    theta6 = mo.ui.slider(-180, 180, step=0.5, value=20, label="phase θ (deg)", show_value=True, full_width=True)
    mode6 = mo.ui.radio(
        options=["use the slider", "inside Θ∂ (centre of the widest band)", "safest admissible phase"],
        value="use the slider",
        label="θ is set by",
        inline=True,
    )
    hue6 = mo.ui.checkbox(value=False, label="colour every point by θ(z)")
    img6 = mo.ui.checkbox(value=True, label="show image singularities")
    mo.vstack([mo.hstack([scen6, depth6], justify="start", gap=2), pot6, theta6, mode6, mo.hstack([hue6, img6], justify="start")])
    return depth6, hue6, img6, mode6, pot6, scen6, theta6


@app.cell(hide_code=True)
def _(
    arc_dist,
    boundary_phases,
    build_obstacle_potential,
    circle_inside,
    crit_in_domain,
    depth6,
    mode6,
    np,
    pot6,
    scen6,
    theta6,
    trace_branch,
    wrap_deg,
):
    _SC = {
        "disk": (2.0, [], [-1 + 0j], [1 + 0j]),
        "one": (2.4, [((0.05, 0.05), 0.55)], [-1.5 + 0.55j, -1.45 - 0.6j], [1.5 - 0.45j, 1.4 + 0.7j]),
        "two": (
            2.7,
            [((-0.15, 0.95), 0.45), ((0.35, -0.85), 0.5)],
            [-1.9 + 1.1j, -2.0 + 0.2j, -1.9 - 0.6j, -1.6 - 1.5j],
            [1.7 + 1.3j, 2.0 + 0.35j, 1.9 - 0.5j, 1.5 - 1.5j],
        ),
    }
    cfg6 = _SC[scen6.value]
    _r0, _obs, _p, _z = cfg6
    P6, circles6 = build_obstacle_potential(_r0, _obs, _p, _z, depth6.value, pot6.value)
    inside6 = circle_inside(_r0, _obs)
    bnd6 = boundary_phases(P6, circles6)
    crit6, thb6 = crit_in_domain(P6, inside6, (-_r0, _r0, -_r0, _r0))
    if mode6.value.startswith("inside"):
        _b = max(bnd6, key=lambda b: b["width"])
        th6 = float(wrap_deg(_b["lo"] + _b["width"] / 2))
    elif mode6.value.startswith("safest"):
        _grid = np.arange(-179.5, 179.6, 0.5)
        _m = [min([arc_dist(t, b) for b in bnd6] + [abs(float(wrap_deg(x - t))) for x in thb6]) for t in _grid]
        th6 = float(_grid[int(np.argmax(_m))])
    else:
        th6 = float(theta6.value)
    branches6 = [trace_branch(P6, i, th6, inside6, _r0) for i in range(len(_p))]
    return P6, bnd6, branches6, cfg6, circles6, crit6, inside6, th6, thb6


@app.cell(hide_code=True)
def _(
    CIRCLE_COL,
    DANGER,
    MUTED,
    P6,
    RC,
    WARN,
    arc_dist,
    bnd6,
    branches6,
    cfg6,
    crit6,
    draw_poles_zeros,
    fig_with_strip,
    fmt_deg,
    hue6,
    hue_background,
    img6,
    in_arc,
    inside6,
    mo,
    np,
    phase_color,
    phase_strip,
    plane_axes,
    plt,
    pot6,
    th6,
    thb6,
    wrap_deg,
):
    _r0, _obs, _p, _z = cfg6
    _fig, _ax, _axs = fig_with_strip()
    _lim = _r0 * 1.1
    plane_axes(_ax, _lim)
    _ax.add_patch(plt.Circle((0, 0), _r0, color="white", zorder=0.5))
    _ax.set_facecolor("#eef1f5")
    if hue6.value:
        hue_background(_ax, P6.theta, _lim, mask=inside6)
    for (_c, _r) in _obs:
        _ax.add_patch(plt.Circle(_c, _r, color="#e3e8ee", zorder=1))
    if img6.value and pot6.value == "mt":
        _iz, _ip = P6.Z[P6.n_orig:], P6.P[P6.n_orig:]
        _ax.plot(_iz.real, _iz.imag, "o", ms=3, mfc="none", mec=MUTED, alpha=0.7, zorder=2)
        _ax.plot(_ip.real, _ip.imag, "x", ms=3.5, color=MUTED, alpha=0.7, zorder=2)
    for _b in bnd6:  # boundary coloured by theta(z)
        _w, _th = _b["w"], wrap_deg(_b["th"])
        _ax.scatter(_w.real, _w.imag, c=phase_color(_th), s=9, marker="o", lw=0, zorder=3)
    for _i, (_pts, _fl, _hit, _end) in enumerate(branches6):
        for _k in range(1, len(_pts)):
            _a, _c2 = _pts[_k - 1], _pts[_k]
            _in = _fl[_k - 1] and _fl[_k]
            _ax.plot([_a.real, _c2.real], [_a.imag, _c2.imag], color=RC[_i % 6], lw=2.6 if _in else 1.4,
                     ls="-" if _in else (0, (3, 3)), alpha=1 if _in else 0.75, zorder=4, solid_capstyle="round")
    for _b in bnd6:  # contact points of the current GRL with the boundary
        _u = wrap_deg(_b["th"] - th6)
        _s = np.where((np.abs(_u[:-1]) < 90) & (np.abs(_u[1:]) < 90) & (np.sign(_u[:-1]) != np.sign(_u[1:])))[0]
        _ax.scatter(_b["w"][_s].real, _b["w"][_s].imag, s=70, color=DANGER, edgecolor="white", lw=1.5, zorder=7)
    for _b, _t in zip(crit6, thb6):
        _near = abs(wrap_deg(_t - th6)) < 3
        _ax.plot(_b.real, _b.imag, "D", ms=7 if _near else 4.5, color=WARN if _near else MUTED, zorder=5)
    draw_poles_zeros(_ax, np.asarray(_p), np.asarray(_z))
    for _j, _b in enumerate(bnd6):
        _C = _b["C"]
        _pos = (_C["r"] * 0.72, _C["r"] * 0.74) if _C["outer"] else (_C["c"].real, _C["c"].imag)
        _ax.text(*_pos, f"$C_{_j}$", color=CIRCLE_COL[_j], ha="center", va="center", fontsize=12, weight="bold", zorder=8)
    _ax.set_title(f"θ = {fmt_deg(th6)}   ·   {'plain potential' if pot6.value == 'plain' else 'reflections'}")
    phase_strip(_axs, th6, bands=[(b, CIRCLE_COL[j]) for j, b in enumerate(bnd6)], ticks=thb6, tick_color=WARN,
                rows_labels=[f"C{j}" for j in range(len(bnd6))])
    _axs.set_title("rows: Θ∂ interval of each circle   ·   ochre ticks: Θb", fontsize=9.5, loc="left", color=MUTED, weight="normal")
    _hits = [b["C"]["name"] for b in bnd6 if in_arc(th6, b)]
    _mb = min([abs(float(wrap_deg(t - th6))) for t in thb6], default=np.inf)
    _md = min(arc_dist(th6, b) for b in bnd6)
    if _hits:
        _status = mo.callout(mo.md(f"**θ ∈ Θ∂**: GRL(θ) touches {', '.join(_hits)}."), kind="danger")
    elif _mb < 0.3:
        _status = mo.callout(mo.md("**θ ∈ Θb**: a break point lies on GRL(θ)."), kind="danger")
    else:
        _status = mo.callout(mo.md(f"**Admissible**: margin {min(_md, _mb):.1f}° from Θ∂ ∪ Θb. All branches stay in D."), kind="success")
    _mt = pot6.value == "mt"
    _hdr = "| circle | excluded interval Θ∂ | width |" + (" residual ε̄ | bound π r ε̄ |" if _mt else "") + "\n|---|---|---|" + ("---|---|" if _mt else "")
    _rows = []
    for _b in bnd6:
        _iv = "all phases" if _b["width"] >= 359.99 else f"[{fmt_deg(_b['lo'])}, {fmt_deg(float(wrap_deg(_b['lo'] + _b['width'])))}]"
        _row = f"| {_b['C']['name']} | {_iv} | {_b['width']:.1f}° |"
        if _mt:
            _row += f" {_b['eps']:.1e} | {_b['bound']:.1f}° |"
        _rows.append(_row)
    _ends = []
    for _i, (_pts, _fl, _hit, _end) in enumerate(branches6):
        if _hit is not None:
            _ends.append(f"robot {_i + 1}: **touches the boundary**")
        elif _end is not None and _end < P6.n_orig:
            _ends.append(f"robot {_i + 1} → $\\zeta_{_end + 1}$")
        else:
            _ends.append(f"robot {_i + 1}: does not reach a goal")
    _out = mo.vstack([_status, mo.as_html(_fig), mo.md(_hdr + "\n" + "\n".join(_rows) + "\n\n" + "  ·  ".join(_ends))])
    plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Reflections, residual and certificate

    **Making circles impenetrable.** The obstacles become impenetrable if the velocity never points into them,
    i.e. if $\partial\Phi/\partial n=0$ on every $C_i$ (homogeneous Neumann condition). By Cauchy–Riemann,
    $\partial\Psi/\partial\tau=\partial\Phi/\partial n$: the condition means that **$\Psi$ is constant on $C_i$**,
    so every circle is a streamline and contributes a **single** boundary phase.

    **Milne–Thomson.** For one circle, $\Omega_C(z)=\Omega(z)+\overline{\Omega(\rho_C(z))}$ with
    $\rho_C(z)=c+r^2/\overline{(z-c)}$ is exact. In the logarithmic setting it only adds an image zero at $\rho_C(\zeta)$
    for each zero and an image pole at $\rho_C(p)$ for each pole, all inside $C$. The $\log(z-c)$ terms cancel in the
    balanced case. With several circles the images perturb the other circles, so the correction is iterated up to
    a depth $m$, with $N_I^{(m)}=2NN_{\rm obs}\sum_{s<m}(N_{\rm obs}-1)^s$ images.

    The truncated $R^{(m)}_{\rm obs}$ is **rational**, with all images outside $\overline D$. So it satisfies the
    Assumption, and **the theorem applies to the computed potential itself**, with no convergence argument.

    **Residual.** For finite $m$, $\Psi$ is only nearly constant on $C_i$. The residual
    $\bar\varepsilon_i=\max_{C_i}|\partial\Phi/\partial n|$ measures how much the velocity field still enters
    obstacle $i$, and it is also the largest slope of $\theta(z)$ along $C_i$.

    **Proposition.**

    1. The interval of $C_i$ has width at most $\pi r_i\bar\varepsilon_i$. A function with slope at most
       $\bar\varepsilon_i$ changes by at most $\pi r_i\bar\varepsilon_i$ over half a turn, which is the longest arc
       between two points of the circle.
    2. **Finite test.** Sample $n_i$ equispaced points on $C_i$, set $h_i=2\pi r_i/n_i$ and
       $L_i=\sum_s 1/\mathrm{dist}(s,C_i)$, where $s$ ranges over all singularities, images included. Since
       $|\Omega'|\le L_i$ on $C_i$, if
       $$\min_\ell d_{2\pi}\big(\Psi(z_{i,\ell}),\theta+\pi\big)>L_ih_i/2 \quad\text{for every } i,$$
       then no point of $\partial D$ is on $\mathrm{GRL}(\theta)$, that is, $\theta\notin\Theta_\partial$.

    The first plot uses the scenario of Section 6 and shows the width of each interval as the depth grows, next to
    its bound.
    """)
    return


@app.cell(hide_code=True)
def _(
    CIRCLE_COL,
    boundary_phases,
    build_obstacle_potential,
    cfg6,
    mo,
    np,
    plt,
):
    _r0, _obs, _p, _z = cfg6
    _ms = [0, 1, 2, 3]
    _W, _B = [], []
    for _m in _ms:
        _P, _C = build_obstacle_potential(_r0, _obs, _p, _z, _m, "plain" if _m == 0 else "mt")
        _bs = boundary_phases(_P, _C)
        _W.append([b["width"] for b in _bs])
        _B.append([b["bound"] if _m > 0 else np.nan for b in _bs])
    _W, _B = np.array(_W), np.array(_B)
    _fig, _ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    for _j in range(_W.shape[1]):
        _ax.semilogy(_ms, np.maximum(_W[:, _j], 1e-6), "o-", color=CIRCLE_COL[_j], lw=2.2, label=f"C{_j}: width of Θ∂ interval")
        _ax.semilogy(_ms[1:], np.maximum(_B[1:, _j], 1e-6), "s--", color=CIRCLE_COL[_j], lw=1.2, alpha=0.7, mfc="white", label=f"C{_j}: bound π r ε̄")
    _ax.set_xticks(_ms)
    _ax.set_xticklabels(["plain", "m = 1", "m = 2", "m = 3"])
    _ax.set_ylabel("degrees (log scale)")
    _ax.grid(True, which="both", color="#eef2f6")
    _ax.legend(frameon=False, fontsize=8.5, ncol=2)
    _ax.set_title("excluded phase interval of each circle vs reflection depth")
    _tbl = "| | " + " | ".join(f"C{j} width / bound" for j in range(_W.shape[1])) + " |\n|---|" + "---|" * _W.shape[1] + "\n"
    for _k, _m in enumerate(_ms):
        _tbl += f"| {'plain' if _m == 0 else 'm = ' + str(_m)} | " + " | ".join(
            f"{_W[_k, j]:.2f}°" + ("" if _m == 0 else f" / {_B[_k, j]:.2f}°") for j in range(_W.shape[1])
        ) + " |\n"
    _out = mo.vstack([mo.as_html(_fig), mo.md(_tbl)])
    plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(circles6, mo):
    circ7 = mo.ui.dropdown(options={c["name"]: i for i, c in enumerate(circles6)}, value=circles6[-1]["name"], label="circle")
    n7 = mo.ui.dropdown(options={str(v): v for v in (16, 32, 64, 128, 256, 512, 1024, 2048)}, value="64", label="samples per circle n")
    mo.vstack(
        [
            mo.md(
                "**The certificate at work.** It uses the potential, depth and phase chosen in Section 6. "
                "The curve is $\\theta(z)$ along the selected circle, the dots are the $n$ samples, and the band of "
                "half-width $L_ih_i/2$ around each sample is where the true curve may go between samples. "
                "The test passes when no band reaches the current $\\theta$ (horizontal line)."
            ),
            mo.hstack([circ7, n7], justify="start", gap=2),
        ]
    )
    return circ7, n7


@app.cell(hide_code=True)
def _(
    CIRCLE_COL,
    DANGER,
    INK,
    OK,
    P6,
    bnd6,
    circ7,
    circles6,
    in_arc,
    mo,
    n7,
    np,
    plt,
    th6,
    wrap_deg,
):
    _rows = []
    _sing = np.concatenate([P6.Z, P6.P])
    _all_pass = True
    _plot = None
    for _i, _C in enumerate(circles6):
        _n = n7.value
        _t = 2 * np.pi * np.arange(_n) / _n
        _w = _C["c"] + _C["r"] * np.exp(1j * _t)
        _h = 2 * np.pi * _C["r"] / _n
        _L = np.sum(1 / np.abs(np.abs(_sing - _C["c"]) - _C["r"]))
        _thr = np.degrees(_L * _h / 2)
        _marg = np.min(np.abs(wrap_deg(P6.theta(_w) - th6)))
        _pass = _marg > _thr
        _all_pass &= _pass
        _true = in_arc(th6, bnd6[_i])
        _rows.append(
            f"| {_C['name']} | {_marg:.2f}° | {_thr:.2f}° | {'pass' if _pass else 'inconclusive'} | {'touches' if _true else 'safe'} |"
        )
        if _i == circ7.value:
            _plot = (_t, P6.theta(_w), _thr, bnd6[_i])
    _t, _ths, _thr, _b = _plot
    _fig, _ax = plt.subplots(figsize=(7.6, 3.4), layout="constrained")
    _ref = th6
    _dense = _b["th"] - 360 * np.round((np.mean(_b["th"]) - _ref) / 360)
    _tt = np.linspace(0, 2 * np.pi, len(_dense))
    _interp = np.interp(_t, _tt, _dense)
    _ys = _ths + 360 * np.round((_interp - _ths) / 360)
    _ax.fill_between(np.degrees(_t), _ys - _thr, _ys + _thr, step="mid", color=CIRCLE_COL[circ7.value], alpha=0.18, lw=0)
    _ax.plot(np.degrees(_tt), _dense, color=CIRCLE_COL[circ7.value], lw=2, label="θ(z) along the circle (dense)")
    _ax.plot(np.degrees(_t), _ys, "o", ms=3.5, color=CIRCLE_COL[circ7.value], label="samples")
    _lo, _hi = min(_dense.min(), (_ys - _thr).min()), max(_dense.max(), (_ys + _thr).max())
    _span = max(_hi - _lo, 12)
    _y0, _y1 = min(_lo, _ref) - 0.2 * _span, max(_hi, _ref) + 0.2 * _span
    for _kk in range(-3, 4):
        _yy = _ref + 360 * _kk
        if _y0 <= _yy <= _y1:
            _ax.axhline(_yy, color=INK, lw=1.4, label="current θ (mod 360°)" if _kk == 0 else None)
    _ax.set_ylim(_y0, _y1)
    _ax.set_xlim(0, 360)
    _ax.set_xlabel("position on the circle (deg)")
    _ax.set_ylabel("θ(z) (deg)")
    _ax.grid(True, color="#eef2f6")
    _ax.legend(frameon=False, fontsize=8.5, loc="best")
    _ax.set_title(f"{circles6[circ7.value]['name']}: certificate band ±L h/2 = ±{_thr:.2f}°")
    _verdict = (
        mo.callout(mo.md(f"**Certificate passes for θ = {th6:.1f}°**: no circle is touched (sufficient condition)."), kind="success")
        if _all_pass
        else mo.callout(
            mo.md("**Certificate inconclusive** for at least one circle: either θ is really in Θ∂, or more samples are needed."),
            kind="warn",
        )
    )
    _out = mo.vstack(
        [
            _verdict,
            mo.as_html(_fig),
            mo.md(
                "| circle | sampled margin | threshold $L_ih_i/2$ | test | truth (dense check) |\n|---|---|---|---|---|\n"
                + "\n".join(_rows)
                + "\n\nThe test is only **sufficient**. When it passes the circle is certainly safe. When it is "
                "inconclusive, increasing $n$ shrinks the threshold like $1/n$."
            ),
        ]
    )
    plt.close(_fig)
    _ = (DANGER, OK)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. The reciprocal root locus as a special case

    The RRL planner uses the upper half disk $D_+=\{|z|<1,\ \Im z>0\}$ and $R_r(z)=G(z)\,G(1/z)$, where $G$ has
    zeros $\zeta_i,\bar\zeta_i$ and poles $p_i,\bar p_i$. The boundary of $D_+$ has two parts:

    - on the unit circle $1/z=\bar z$, so $R_r=|G|^2>0$;
    - on $(-1,1)$ every conjugate pair gives a positive factor.

    So $\Psi_r\equiv0$ on $\partial D_+$ and $\Theta_\partial=\{-\pi\}$. The theorem therefore applies to the
    classical RRL ($\theta=0$) whenever $0\notin\Theta_b$. Confinement to $D_+$, convergence, assignment and
    non-intersection all follow at once. The figure computes $\Theta_\partial$ numerically from samples of
    $\partial D_+$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    theta8 = mo.ui.slider(-179, 179, step=1, value=0, label="phase θ (deg), 0 = classical RRL", show_value=True, full_width=True)
    theta8
    return (theta8,)


@app.cell(hide_code=True)
def _(
    MUTED,
    Pot,
    RC,
    WARN,
    boundary_phases,
    crit_in_domain,
    draw_poles_zeros,
    fig_with_strip,
    fmt_deg,
    mo,
    np,
    phase_color,
    phase_strip,
    plane_axes,
    plt,
    theta8,
    trace_branch,
    wrap_deg,
):
    _p = np.array([-0.75 + 0.2j, -0.45 + 0.55j, -0.1 + 0.25j])
    _z = np.array([0.7 + 0.35j, 0.3 + 0.8j, 0.45 + 0.15j])
    _Z = np.concatenate([_z, np.conj(_z), 1 / _z, 1 / np.conj(_z)])
    _P = np.concatenate([_p, np.conj(_p), 1 / _p, 1 / np.conj(_p)])
    _pot = Pot(_Z, _P, 0.0, 3)

    def _inside(w, m=0.0):
        w = np.asarray(w, complex)
        return (np.abs(w) < 1 - m) & (w.imag > m)

    _s = np.linspace(0, np.pi, 900)
    _curve = np.concatenate([np.exp(1j * _s), np.linspace(-1, 1, 900)[1:]]) * (1 - 1e-9) + 1e-9j
    _bnd = boundary_phases(_pot, [dict(w=_curve, name="∂D₊")])
    _crit, _thb = crit_in_domain(_pot, _inside, (-1, 1, 0, 1))
    _th = float(theta8.value)
    _br = [trace_branch(_pot, i, _th, _inside, 1.0) for i in range(3)]
    _fig, _ax, _axs = fig_with_strip(size=(7.2, 6.4), ratio=(4.4, 1))
    plane_axes(_ax, 1.12, 0.45j)
    _ax.add_patch(plt.matplotlib.patches.Wedge((0, 0), 1, 0, 180, color="white", zorder=0.5))
    _ax.set_facecolor("#eef1f5")
    _ax.scatter(_curve.real, _curve.imag, c=phase_color(wrap_deg(_bnd[0]["th"])), s=8, lw=0, zorder=3)
    for _i, (_pts, _fl, _hit, _end) in enumerate(_br):
        _ax.plot(_pts.real, _pts.imag, color=RC[_i], lw=2.6, zorder=4)
    _ax.plot(_crit.real, _crit.imag, "D", color=MUTED, ms=5, zorder=5)
    draw_poles_zeros(_ax, _p, _z)
    _ax.set_title(f"RRL in the upper half disk, θ = {fmt_deg(_th)}")
    phase_strip(_axs, _th, bands=[(_bnd[0], "#5a6879")], ticks=_thb, tick_color=WARN, rows_labels=["∂D₊"])
    _ends = ", ".join(
        (f"$p_{i + 1}\\to\\zeta_{e + 1}$" if (e is not None and e < 3 and h is None) else f"robot {i + 1}: leaves $D_+$")
        for i, (_, _, h, e) in enumerate(_br)
    )
    _out = mo.vstack(
        [
            mo.as_html(_fig),
            mo.md(
                f"Numerical $\\Theta_\\partial$: starts at {fmt_deg(_bnd[0]['lo'])}, width **{_bnd[0]['width']:.1e}°**. "
                f"It is the single phase $\\pm180^\\circ$, so the boundary of $D_+$ is one streamline.  \n"
                f"Critical phases in $D_+$: {', '.join(fmt_deg(t) for t in np.sort(_thb)) or 'none'}.  \nAssignment: {_ends}."
            ),
        ]
    )
    plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Computational procedure

    ```text
    Algorithm 1 — Certified GRL planning
    Input : inflated polygonal workspace P, starts, goals, depth m
    1  compute the conformal map f : P -> D and the poles p_i, zeros ζ_i in D
    2  generate the images up to depth m and form R_obs^(m)
    3  compute the zeros of Ω' in D (Newton from a grid);
       verify their number by the argument principle
    4  form Θb = {π − arg R(b)}; test the certificate on a grid of phases
    5  select θ maximising the distance from Θb and the certificate margin
    6  trace the N branches of GRL(θ) from the departure directions; map back by f^{-1}
    Output: paths, assignment σ_θ, margin — or failure if no phase passes
    ```

    **Departure direction.** Near $p_i$, $R(z)\simeq\rho_i/(z-p_i)$ with residue $\rho_i$, so the branch leaves
    $p_i$ along $\arg(z-p_i)=\pi+\theta+\arg\rho_i$. The tracer used in Sections 6 and 8 starts there and follows
    $\dot z=\overline{\Omega'}/|\Omega'|$. After every step a Newton correction puts the point back on $\Psi=\theta+\pi$.

    **Completeness of step 3.** $\Omega'$ has exactly $2N$ simple poles in $D$, the original starts and goals. By
    the argument principle,
    $$\#\{\Omega'=0 \text{ in } D\}=2N+\frac{1}{2\pi j}\oint_{\partial D}\frac{\Omega''}{\Omega'}\,dz ,$$
    with the outer circle counter-clockwise and the obstacles clockwise. The table checks this count against the
    Newton search for the scenario selected in Section 6.
    """)
    return


@app.cell(hide_code=True)
def _(P6, argument_principle_count, circles6, crit6, mo):
    _cnt = argument_principle_count(P6, circles6, P6.n_orig)
    _ok = abs(round(_cnt) - len(crit6)) == 0 and abs(_cnt - round(_cnt)) < 1e-6
    mo.vstack(
        [
            mo.md(
                f"| quantity | value |\n|---|---|\n"
                f"| singularities of $R$ (originals + images) | {len(P6.Z) + len(P6.P)} |\n"
                f"| critical points found by Newton in $D$ | **{len(crit6)}** |\n"
                f"| argument-principle count | **{_cnt:.6f}** |"
            ),
            mo.callout(
                mo.md(
                    "The two counts agree, so the critical set used to build Θb is complete."
                    if _ok
                    else "The counts differ: refine the Newton grid (a critical point was missed or a zero lies close to ∂D)."
                ),
                kind="success" if _ok else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - The GRL is the preimage of the negative real axis under $F_\theta=e^{j\theta}R$, and through every point passes
      exactly one GRL.
    - Two sets of phases are excluded. **$\Theta_b$** is finite and contains the phases whose GRL passes through a
      break point. **$\Theta_\partial$** has one interval per boundary circle and contains the phases whose GRL
      touches the boundary.
    - For every other phase the theorem gives $N$ disjoint branches, convergence to distinct goals with an induced
      assignment, safety with respect to robots and obstacles, and synchronous arrival in finite time when the
      gain is used as a clock.
    - For finite reflection depth, the boundary intervals are bounded by $\pi r_i\bar\varepsilon_i$, and safety is
      certified by a finite test on the potential that is actually computed.

    *All computations in this notebook run in your browser with NumPy and Matplotlib. The numerical routines are
    in the hidden cells at the top.*
    """)
    return


if __name__ == "__main__":
    app.run()
