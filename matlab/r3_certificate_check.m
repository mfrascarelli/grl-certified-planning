%R3_CERTIFICATE_CHECK  Run after an RLP_PFS_* script, with in the workspace:
%   rob_i_z, rob_g_z  (starts/goals in the canonical domain D)
%   centers, radii    (circles of D, from the PlgCirMap object f: f.cent, f.rad)
% Optional: n_inv (reflection depth, default 3), theta_used (rad).
if ~exist('n_inv','var'), n_inv = 3; end
N = numel(rob_i_z);
for m = 1:n_inv
    [P, Z, info] = grl_images(rob_i_z, rob_g_z, centers, radii, m);
    tic; S = grl_phase_margins(P, Z, N, centers, radii); tS = toc;
    fprintf('\n=== depth m = %d : %d images per type, %.1f s\n', m, info.NI, tS);
    fprintf('critical points: %d distinct, %d with multiplicity, argument principle predicts %d\n', numel(S.b), S.ncrit_found, S.ncrit_expected);
    if S.ncrit_found ~= S.ncrit_expected, warning('Critical set possibly incomplete: Theta_b may miss phases.'); end
    for i = 1:numel(radii)
        C = S.circle(i);
        fprintf('  C%-2d  eps_max=%.2e  osc(Psi)=%.2e rad  bound pi*r*eps=%.2e  winding=%d  slack=%.1e\n', ...
            i, C.eps_max, C.osc_psi, pi*radii(i)*C.eps_max, C.winding, C.slack);
    end
    fprintf('certified fraction of phases: %.1f%%   best theta = %.4f rad (margin %.3e rad)\n', ...
        100*S.frac_certified, S.best_theta, S.best_margin);
    if exist('theta_used','var')
        [~, k] = min(abs(angle(exp(1i*(S.theta - theta_used)))));
        fprintf('theta_used = %.4f: margin %.3e rad (crit %.3e, cert %.3e)\n', theta_used, ...
            S.margin(k), S.margin_crit(k), S.margin_cert(k));
    end
    R3(m) = S; %#ok<SAGROW>
end
figure; plot(R3(end).theta, R3(end).margin, 'b-'); hold on; plot([-pi pi], [0 0], 'k--');
xlabel('\theta [rad]'); ylabel('margin [rad]'); title(sprintf('Certificate margin, m = %d', n_inv)); grid on;
