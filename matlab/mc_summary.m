function T = mc_summary(files)
%MC_SUMMARY Aggregate Monte Carlo results and print a LaTeX table.
if nargin < 1, files = {'mc_flagship.mat','mc_warehouse.mat'}; end
fprintf('%% domain & M & GRL cert. & GRL succ. & crit.ok & margin(med) & len/LB & minsep & time | HFM ok & cross & len/LB & time | APF succ & coll & pen & len/LB\n');
for f = 1:numel(files)
    S = load(files{f}); R = S.R; M = numel(R);
    v = @(x) [R.(x)];
    T(f).file = files{f}; T(f).M = M;
    T(f).grl_cert = mean(v('grl_certified')); T(f).grl_succ = mean(v('grl_success'));
    T(f).crit_ok = mean(v('grl_crit_ok')); T(f).margin = median(v('grl_margin'));
    T(f).grl_len = median(v('grl_len')./v('lb_len')); T(f).grl_minsep = median(v('grl_minsep'));
    T(f).grl_time = median(v('grl_time'));
    T(f).hfm_ok = mean(v('hfm_ok')); T(f).hfm_cross = mean(v('hfm_cross') > 0);
    T(f).hfm_len = median(v('hfm_len')./v('lb_len')); T(f).hfm_time = median(v('hfm_time'));
    T(f).apf_succ = mean(v('apf_success')); T(f).apf_coll = mean(v('apf_collision'));
    T(f).apf_pen = mean(v('apf_penetration'));
    ok = v('apf_success') > 0; al = v('apf_len'); lb = v('lb_len'); T(f).apf_len = median(al(ok) ./ lb(ok));
    fprintf('%s & %d & %.0f\\%% & %.0f\\%% & %.0f\\%% & %.2f & %.2f & %.3f & %.1f & %.0f\\%% & %.0f\\%% & %.2f & %.1f & %.0f\\%% & %.0f\\%% & %.0f\\%% & %.2f \\\\\n', ...
        files{f}, M, 100*T(f).grl_cert, 100*T(f).grl_succ, 100*T(f).crit_ok, T(f).margin, T(f).grl_len, ...
        T(f).grl_minsep, T(f).grl_time, 100*T(f).hfm_ok, 100*T(f).hfm_cross, T(f).hfm_len, T(f).hfm_time, ...
        100*T(f).apf_succ, 100*T(f).apf_coll, 100*T(f).apf_pen, T(f).apf_len);
end
end
