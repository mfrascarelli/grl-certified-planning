function out = bl_hfm(s, g, perm, centers, radii, m)
%BL_HFM Independent harmonic-function planning: one robot at a time.
%   Robot i uses the single-pair harmonic potential (source s_i, sink g_perm(i),
%   Neumann on all circles via the same images, depth m) and follows the
%   shortest of 8 streamlines. No coordination among robots.
N = numel(s); th = linspace(-pi, pi, 9); th(end) = [];
for i = 1:N
    [P, Z] = grl_images(s(i), g(perm(i)), centers, radii, m);
    bestL = inf; best = [];
    for t = th
        B = grl_trace_branch(P, Z, 1, 1, t, centers, radii, struct('maxit', 4000));
        if B.inD && B.goal == 1 && B.length < bestL, bestL = B.length; best = B; end
    end
    paths{i} = best; %#ok<AGROW>
end
out.ok = all(~cellfun(@isempty, paths));
out.crossings = 0; out.length = nan(1,N);
if ~out.ok, return; end
for i = 1:N, out.length(i) = paths{i}.length; end
for i = 1:N, for j = i+1:N
    out.crossings = out.crossings + count_cross(paths{i}.z, paths{j}.z);
end, end
out.paths = paths;
end

function n = count_cross(a, b)
% number of proper intersections between polylines a and b (complex)
n = 0;
a1 = a(1:end-1); a2 = a(2:end);
for k = 1:numel(b)-1
    b1 = b(k); b2 = b(k+1);
    % bounding box prefilter
    if max(real(b1),real(b2)) < min(min(real(a1),real(a2))), continue; end
    d1 = cr(b2-b1, a1-b1); d2 = cr(b2-b1, a2-b1);
    d3 = cr(a2-a1, b1-a1); d4 = cr(a2-a1, b2-a1);
    n = n + sum((d1.*d2 < 0) & (d3.*d4 < 0));
end
end
function c = cr(u, v), c = real(u).*imag(v) - imag(u).*real(v); end
