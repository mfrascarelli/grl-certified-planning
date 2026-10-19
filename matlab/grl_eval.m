function [dO, ddO, psi] = grl_eval(z, P, Z)
%GRL_EVAL Omega = -log R = sum log(z-P) - sum log(z-Z).
%   dO = Omega'(z), ddO = Omega''(z), psi = Psi(z) wrapped to (-pi,pi].
%   Gradient of Phi as complex number: conj(dO).
sz = size(z); z = z(:).';
dO = zeros(size(z)); ddO = dO; psi = dO;
B = 200; % chunk over singularities to limit memory
for k = 1:B:numel(P)
    w = z - P(k:min(k+B-1,end));            % (nP x nz)
    dO = dO + sum(1./w,1); ddO = ddO - sum(1./w.^2,1); psi = psi + sum(angle(w),1);
end
for k = 1:B:numel(Z)
    w = z - Z(k:min(k+B-1,end));
    dO = dO - sum(1./w,1); ddO = ddO + sum(1./w.^2,1); psi = psi - sum(angle(w),1);
end
psi = angle(exp(1i*psi));
dO = reshape(dO,sz); ddO = reshape(ddO,sz); psi = reshape(psi,sz);
end
