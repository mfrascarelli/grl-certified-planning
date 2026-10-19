function out = grl_plan(P, Z, N, theta, centers, radii, opts)
%GRL_PLAN Trace all N branches of GRL(theta) and check Theorem items 1-3.
if nargin < 7, opts = struct; end
[io, in] = grl_domain(centers, radii);
for i = 1:N, br(i) = grl_trace_branch(P, Z, N, i, theta, centers, radii, opts); end %#ok<AGROW>
out.br = br; out.sigma = [br.goal];
out.all_reached = all(out.sigma > 0) && numel(unique(out.sigma)) == N;
out.all_inD = all([br.inD]);
% min distance between distinct paths (geometric) and to obstacles
dmin = inf;
for i = 1:N, for j = i+1:N
    zi = br(i).z(2:end-1); zj = br(j).z(2:end-1);
    dmin = min(dmin, min(min(abs(zi - zj.'))));
end, end
out.min_sep = dmin;
cl = inf;
for i = 1:N
    zz = br(i).z;
    cl = min(cl, min(radii(io) - abs(zz - centers(io))));
    for c = in, cl = min(cl, min(abs(zz - centers(c)) - radii(c))); end
end
out.min_clear = cl; out.psidrift = max([br.psidrift]); out.length = [br.length];
end
