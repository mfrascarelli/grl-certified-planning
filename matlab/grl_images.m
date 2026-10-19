function [P, Z, info] = grl_images(poles, zeros_, centers, radii, m)
%GRL_IMAGES Original singularities plus successive-reflection images.
%   [P,Z] = GRL_IMAGES(poles, zeros, centers, radii, m) returns all poles P
%   and zeros Z of R_obs^(m) (originals first). Same recursion as
%   computeOmega_obs_IM (reflect previous images in every other circle).
%   Center terms log(z-c) cancel in the balanced case and are omitted.
poles = poles(:); zeros_ = zeros_(:); centers = centers(:); radii = radii(:);
if numel(poles) ~= numel(zeros_), error('Balanced case required.'); end
Nc = numel(centers);
prevP = cell(Nc,1); prevZ = cell(Nc,1);
P = poles; Z = zeros_;
for s = 1:m
    curP = cell(Nc,1); curZ = cell(Nc,1);
    for i = 1:Nc
        c = centers(i); r = radii(i);
        if s == 1
            srcP = poles; srcZ = zeros_;
        else
            srcP = vertcat(prevP{[1:i-1, i+1:Nc]});
            srcZ = vertcat(prevZ{[1:i-1, i+1:Nc]});
        end
        curP{i} = c + r^2 ./ conj(srcP - c);
        curZ{i} = c + r^2 ./ conj(srcZ - c);
    end
    P = [P; vertcat(curP{:})]; Z = [Z; vertcat(curZ{:})];
    prevP = curP; prevZ = curZ;
end
info.N = numel(poles); info.NI = numel(P) - numel(poles);
end
