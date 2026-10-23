function out = bl_apf(s, g, perm, centers, radii, opts)
%BL_APF Multi-robot artificial potential field (Khatib) in the canonical domain.
%   U_i = ka/2 |z_i - g_i|^2 + sum_obstacles kr/2 (1/rho - 1/rho0)^2 (rho<rho0)
%         + sum_j kr/2 (1/d_ij - 1/d0)^2 (d_ij<d0).  Normalized gradient steps.
if nargin < 6, opts = struct; end
ka = 1; kr = 0.002; rho0 = 0.08; d0 = 0.08; h = 2e-3; T = 4000; tol = 5e-3;
[io, in] = grl_domain(centers, radii);
z = s(:); gg = g(perm); gg = gg(:); N = numel(z);
minsep = inf; minclear = inf; L = zeros(N,1); stall = false(N,1); done = false(N,1);
for it = 1:T
    F = -ka*(z - gg);
    for i = 1:N
        % obstacles (inner circles + outer)
        for c = in
            v = z(i) - centers(c); rho = abs(v) - radii(c);
            if rho < rho0, F(i) = F(i) + kr*(1/rho - 1/rho0)/rho^2 * v/abs(v); end
        end
        v = z(i) - centers(io); rho = radii(io) - abs(v);
        if rho < rho0, F(i) = F(i) - kr*(1/rho - 1/rho0)/rho^2 * v/abs(v); end
        for j = [1:i-1, i+1:N]
            v = z(i) - z(j); dd = abs(v);
            if dd < d0, F(i) = F(i) + kr*(1/dd - 1/d0)/dd^2 * v/dd; end
        end
    end
    F(done) = 0;
    step = h * F ./ max(abs(F), 1e-12);
    step(abs(F) < 1e-6) = 0;
    zn = z + step; L = L + abs(zn - z); z = zn;
    done = done | abs(z - gg) < tol;
    D = abs(z - z.'); D(1:N+1:end) = inf; minsep = min(minsep, min(D(:)));
    cl = radii(io) - abs(z - centers(io));
    for c = in, cl = min(cl, abs(z - centers(c)) - radii(c)); end
    minclear = min(minclear, min(cl));
    if all(done), break; end
end
% count the final segment to the goal, so that L is the length of a path that
% ends exactly at the goal (the loop stops within tol of it)
L(done) = L(done) + abs(z(done) - gg(done));
out.success = all(done); out.reached = sum(done); out.minsep = minsep;
out.minclear = minclear; out.length = L.'; out.iters = it;
out.collision = minsep < 0.01; out.penetration = minclear < 0;
end
