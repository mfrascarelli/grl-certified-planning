function R = mc_canonical(name, inputsfile, N, M, m, seed)
%MC_CANONICAL Monte Carlo in the canonical circular domain (no conformal map).
%   M random instances of N starts/goals in D; GRL (certified phase) vs
%   independent HFM and multi-robot APF, both with min-distance assignment.
S0 = load(inputsfile); centers = S0.centers; radii = S0.radii;
[io, in] = grl_domain(centers, radii);
rand('state', seed);
clr = 0.04; sep = 0.08;
R = struct([]);
for k = 1:M
    pts = [];
    while numel(pts) < 2*N
        z = centers(io) + radii(io)*sqrt(rand)*exp(2i*pi*rand);
        ok = radii(io) - abs(z - centers(io)) > clr && all(abs(z - centers(in)) - radii(in) > clr);
        if ok && (isempty(pts) || min(abs(pts - z)) > sep), pts(end+1) = z; end %#ok<AGROW>
    end
    s = pts(1:N); g = pts(N+1:end);
    r = struct('k', k);
    % --- GRL
    tic;
    [P, Z] = grl_images(s, g, centers, radii, m);
    Sm = grl_phase_margins(P, Z, N, centers, radii, struct('ntheta', 720, 'ngrid', 60, 'tolcert', 1e-2, 'nmax', 5e4));
    r.grl_crit_ok = (Sm.ncrit_found == Sm.ncrit_expected);
    r.grl_margin = Sm.best_margin; r.grl_certified = Sm.best_margin > 0;
    r.grl_frac = Sm.frac_certified;
    o = grl_plan(P, Z, N, Sm.best_theta, centers, radii);
    r.grl_time = toc;
    r.grl_reached = o.all_reached; r.grl_inD = o.all_inD; r.grl_minsep = o.min_sep;
    r.grl_minclear = o.min_clear; r.grl_len = sum(o.length);
    r.grl_success = r.grl_certified && o.all_reached && o.all_inD;
    % --- straight-line reference length (lower bound ignoring obstacles)
    perm = bl_assign(s, g);
    r.lb_len = sum(abs(s - g(perm)));
    % --- HFM independent
    tic; h = bl_hfm(s, g, perm, centers, radii, m); r.hfm_time = toc;
    r.hfm_ok = h.ok; r.hfm_cross = h.crossings; r.hfm_len = sum(h.length);
    % --- APF
    tic; a = bl_apf(s, g, perm, centers, radii); r.apf_time = toc;
    r.apf_success = a.success; r.apf_reached = a.reached; r.apf_minsep = a.minsep;
    r.apf_collision = a.collision; r.apf_penetration = a.penetration; r.apf_len = sum(a.length);
    R = [R, r]; %#ok<AGROW>
    fprintf('%s %3d: GRL cert=%d succ=%d marg=%.2f | HFM cross=%d | APF succ=%d coll=%d pen=%d\n', name, k, ...
        r.grl_certified, r.grl_success, r.grl_margin, r.hfm_cross, r.apf_success, r.apf_collision, r.apf_penetration);
    save(sprintf('mc_%s.mat', name), 'R', '-v7');
end
end
