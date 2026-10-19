function B = grl_trace_branch(P, Z, N, i, theta, centers, radii, opts)
%GRL_TRACE_BRANCH Follow the GRL(theta) branch leaving pole P(i).
%   Points satisfy Omega(z) = Phi + j(theta+pi), i.e. 1 + k e^{j theta} R(z) = 0
%   with k = e^Phi (Lemma). Parameter: Phi (log-gain). Predictor dz = dPhi/Omega',
%   holomorphic Newton corrector on Omega(z) = target (Psi wrapped mod 2pi).
%   Output B.z (path), B.k (gain), B.t (synchronous time t = 1/k0 - 1/k),
%   B.goal (index of the zero reached; 0 if none / outside D), B.inD (stays in D),
%   B.psidrift (max |Psi - (theta+pi)|).
if nargin < 8, opts = struct; end
d = @(f,v) opts_get(opts, f, v);
hmax = d('hmax', 2e-3*max(radii)); rstart = d('rstart', 1e-4*max(radii));
rgoal = d('rgoal', 1e-4*max(radii)); maxit = d('maxit', 2e5);
[io, in] = grl_domain(centers, radii);
inD = @(z) abs(z - centers(io)) < radii(io) & all(abs(z - centers(in)) > radii(in));
psi0 = angle(exp(1i*(theta + pi)));
p = P(i); Pr = P([1:i-1, i+1:end]);
% regular part of Omega at p: Omega ~ log(z-p) + c
[c] = grl_omega_reg(p, Pr, Z);
w0 = rstart * exp(1i*(psi0 - imag(c)));          % arg(z-p) = psi0 - Im c
z = p + w0;
phi = real(c) + log(rstart);
[z] = correct(z, phi, psi0, P, Z);
zs = z; ph = phi; goal = 0; ok = true;
for it = 1:maxit
    [~, dO] = grl_omega(z, P, Z);
    dphi = hmax * abs(dO);                         % |dz| ~ hmax
    zp = z + dphi ./ dO;                           % predictor (dOmega = dphi real)
    [zn, res] = correct(zp, phi + dphi, psi0, P, Z);
    if abs(zn - zp) > 0.5*hmax || res > 1e-9
        hmax = hmax/2; if hmax < 1e-12, break; end, continue;
    end
    z = zn; phi = phi + dphi; zs(end+1,1) = z; ph(end+1,1) = phi; %#ok<AGROW>
    if ~inD(z), ok = false; end
    [dz, j] = min(abs(Z - z));
    if dz < rgoal, goal = j; break; end
    if abs(z) > 10*max(radii) + max(abs(centers)), break; end
    hmax = min(hmax*1.2, d('hmax', 2e-3*max(radii)));
end
if goal > 0, zs(end+1,1) = Z(goal); ph(end+1,1) = Inf; end
[Om] = grl_omega(zs(1:end-(goal>0)), P, Z);
B.z = zs; B.k = exp(ph); B.goal = goal * (goal <= N); B.goal_any = goal;
B.inD = ok && goal <= N && goal > 0;
B.psidrift = max(abs(angle(exp(1i*(imag(Om) - psi0)))));
k0 = B.k(1); B.t = 1/k0 - 1./B.k;
B.length = sum(abs(diff(zs)));
end

function [z, res] = correct(z, phi, psi0, P, Z)
for c = 1:8
    [Om, dO] = grl_omega(z, P, Z);
    r = (real(Om) - phi) + 1i*angle(exp(1i*(imag(Om) - psi0)));
    z = z - r / dO;
    if abs(r) < 1e-12, break; end
end
[Om] = grl_omega(z, P, Z);
res = abs((real(Om) - phi) + 1i*angle(exp(1i*(imag(Om) - psi0))));
end

function c = grl_omega_reg(p, Pr, Z)
c = sum(log(p - Pr)) - sum(log(p - Z));
c = real(c) + 1i*angle(exp(1i*imag(c)));
end

function v = opts_get(o, f, v)
if isfield(o, f), v = o.(f); end
end
