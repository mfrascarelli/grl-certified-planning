function [Om, dO] = grl_omega(z, P, Z)
%GRL_OMEGA Omega(z) = sum log(z-P) - sum log(z-Z) (Psi wrapped) and Omega'(z).
sz = size(z); z = z(:).';
phi = zeros(size(z)); psi = phi; dO = phi; B = 200;
for k = 1:B:numel(P)
    w = z - P(k:min(k+B-1,end));
    phi = phi + sum(log(abs(w)),1); psi = psi + sum(angle(w),1); dO = dO + sum(1./w,1);
end
for k = 1:B:numel(Z)
    w = z - Z(k:min(k+B-1,end));
    phi = phi - sum(log(abs(w)),1); psi = psi - sum(angle(w),1); dO = dO - sum(1./w,1);
end
Om = reshape(phi + 1i*angle(exp(1i*psi)), sz); dO = reshape(dO, sz);
end
