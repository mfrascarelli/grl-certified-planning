function S = grl_phase_margins(P, Z, N, centers, radii, opts)
%GRL_PHASE_MARGINS Critical phases, boundary phases and certificate.
%   Theorem (certified coordinated GRL motion) needs theta not in
%   Theta_b (critical) and not in Theta_d (boundary). Prop. (certificate):
%   theta notin Theta_d if, on every circle C_i,
%       min_l d2pi(Psi(z_il), theta+pi) > L_i h_i / 2.
%   S.margin(theta) = min( dist to Theta_b , certificate margin ) [rad];
%   S.margin > 0  ==> theta certified admissible.
if nargin < 6, opts = struct; end
if ~isfield(opts,'ntheta'), opts.ntheta = 3600; end
if ~isfield(opts,'tolcert'), opts.tolcert = 1e-3; end   % target L h/2
if ~isfield(opts,'nmax'), opts.nmax = 4e5; end
if ~isfield(opts,'ngrid'), opts.ngrid = 200; end
d2pi = @(a,b) abs(angle(exp(1i*(a-b))));
th = linspace(-pi, pi, opts.ntheta+1); th(end) = [];
% critical phases: theta = Psi(b) - pi
[b, nexp, nfound, mult] = grl_critical_points(P, Z, N, centers, radii, opts.ngrid);
[~, ~, psib] = grl_eval(b, P, Z);
S.b = b; S.theta_b = angle(exp(1i*(psib - pi))); S.ncrit_expected = nexp; S.ncrit_found = nfound; S.mult = mult;
mb = inf(size(th));
for k = 1:numel(S.theta_b), mb = min(mb, d2pi(th, S.theta_b(k))); end
% boundary circles
allS = [P; Z]; mc = inf(size(th));
for i = 1:numel(radii)
    c = centers(i); r = radii(i);
    L = sum(1 ./ abs(abs(allS - c) - r));
    n = min(opts.nmax, max(4000, ceil(pi*r*L/opts.tolcert)));
    h = 2*pi*r/n; t = (0:n-1)*2*pi/n; zc = c + r*exp(1i*t);
    [dO, ~, psic] = grl_eval(zc, P, Z);
    nrm = exp(1i*t);
    epsn = abs(real(dO .* nrm));                    % |dPhi/dn| on C_i
    u = unwrap(psic);
    S.circle(i).L = L; S.circle(i).n = n; S.circle(i).slack = L*h/2;
    S.circle(i).eps_max = max(epsn); S.circle(i).osc_psi = max(u) - min(u);
    S.circle(i).winding = round((u(end) - u(1) + angle(exp(1i*(psic(1)-psic(end)))))/(2*pi));
    % distance from theta+pi to the sampled values of Psi on C_i
    ps = unique(psic(:)).'; dmin = inf(size(th));
    tg = angle(exp(1i*(th + pi)));
    idx = interp1([ps-2*pi, ps, ps+2*pi], 1:3*numel(ps), tg, 'nearest');
    ext = [ps-2*pi, ps, ps+2*pi];
    for s = -1:1
        j = min(max(idx+s,1), numel(ext));
        dmin = min(dmin, abs(tg - ext(j)));
    end
    mc = min(mc, dmin - L*h/2);
end
S.theta = th; S.margin_crit = mb; S.margin_cert = mc;
S.margin = min(mb, mc);
[S.best_margin, k] = max(S.margin); S.best_theta = th(k);
S.frac_certified = mean(S.margin > 0);
end
