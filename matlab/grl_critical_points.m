function [b, nexp, nfound, mult] = grl_critical_points(P, Z, N, centers, radii, ngrid)
%GRL_CRITICAL_POINTS Zeros of Omega' in D, with argument-principle check.
%   b: critical points found; nexp: number predicted by the argument
%   principle (#zeros = 2N + winding of Omega' along dD, with multiplicity);
%   mult: multiplicity of each point (local winding of Omega');
%   nfound = sum(mult), to be compared with nexp.
%   Newton multistart (grid + rings near circles and singularities), then
%   deflated Newton until the count matches the argument principle.
if nargin < 6, ngrid = 200; end
[io, in] = grl_domain(centers, radii);
c0 = centers(io); r0 = radii(io);
inD = @(z) abs(z - c0) < r0 & all(abs(z(:) - centers(in).') > radii(in).', 2);
sing = [P(1:N); Z(1:N)];
% --- argument principle (outer ccw, inner cw)
wind = 0; nb = 40000; t = linspace(0, 2*pi, nb+1);
for i = 1:numel(radii)
    zb = centers(i) + radii(i)*exp(1i*t);
    d = grl_eval(zb, P, Z);
    dw = sum(angle(d(2:end)./d(1:end-1))) / (2*pi);
    if i == io, wind = wind + dw; else, wind = wind - dw; end
end
nexp = 2*N + round(wind);
% --- seeds: grid in D + rings around circles and singularities
[X, Y] = meshgrid(linspace(real(c0)-r0, real(c0)+r0, ngrid), linspace(imag(c0)-r0, imag(c0)+r0, ngrid));
z0 = X(:) + 1i*Y(:);
a = 2*pi*(0:63).'/64;
for i = 1:numel(radii)
    for f = [0.98 1.02 1.1 1.3]
        z0 = [z0; centers(i) + f*radii(i)*exp(1i*a)]; %#ok<AGROW>
    end
end
for k = 1:numel(sing)
    for rr = r0*[0.003 0.01 0.03 0.1]
        z0 = [z0; sing(k) + rr*exp(1i*a)]; %#ok<AGROW>
    end
end
z0 = z0(inD(z0));
z0 = z0(min(abs(z0 - sing.'), [], 2) > 1e-6*r0);
b = newton_pass(z0, [], P, Z);
b = keep(b, inD, sing, r0, [], P, Z);
% --- deflation rounds if some zeros are still missing
for round_ = 1:5
    mult = local_mult(b, P, Z, sing, centers, radii);
    if sum(mult) >= nexp, break; end
    bn = newton_pass(z0, b, P, Z);
    b = keep(bn, inD, sing, r0, b, P, Z);
end
mult = local_mult(b, P, Z, sing, centers, radii);
nfound = sum(mult);
end

function z = newton_pass(z, defl, P, Z)
for it = 1:80
    [d, dd] = grl_eval(z, P, Z);
    q = dd ./ d;
    for k = 1:numel(defl), q = q - 1 ./ (z - defl(k)); end
    step = 1 ./ q;
    step(abs(step) > 0.1) = 0.1 * step(abs(step) > 0.1) ./ abs(step(abs(step) > 0.1)); % damping
    z = z - step;
    z(~isfinite(z)) = NaN;
end
z = z(isfinite(z));
end

function b = keep(z, inD, sing, r0, b, P, Z)
if isempty(z), return; end
d = grl_eval(z, P, Z);
ok = abs(d) < 1e-8 & inD(z) & min(abs(z - sing.'), [], 2) > 1e-9*r0;
z = z(ok);
for k = 1:numel(z)
    if isempty(b) || min(abs(b - z(k))) > 1e-7, b = [b; z(k)]; end %#ok<AGROW>
end
end

function mult = local_mult(b, P, Z, sing, centers, radii)
% multiplicity of each critical point = winding of Omega' on a small circle
mult = zeros(size(b)); a = linspace(0, 2*pi, 2001);
for k = 1:numel(b)
    others = b([1:k-1, k+1:end]);
    dc = min(abs(abs(b(k) - centers) - radii));
    rho = 0.3 * min([abs(others - b(k)); abs(sing - b(k)); dc]);
    d = grl_eval(b(k) + rho*exp(1i*a), P, Z);
    mult(k) = round(sum(angle(d(2:end)./d(1:end-1))) / (2*pi));
end
end
