function perm = bl_assign(s, g)
%BL_ASSIGN Min-sum-of-squared-distances assignment (brute force, N<=7).
N = numel(s); Pm = perms(1:N); best = inf;
for k = 1:size(Pm,1)
    c = sum(abs(s(:) - g(Pm(k,:)).').^2);
    if c < best, best = c; perm = Pm(k,:); end
end
end
