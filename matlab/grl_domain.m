function [iout, inner] = grl_domain(centers, radii)
%GRL_DOMAIN Index of the outer circle (largest radius) and of inner ones.
[~, iout] = max(radii); inner = setdiff(1:numel(radii), iout);
end
