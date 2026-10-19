# MATLAB implementation

Code for Theorem 1 ("Certified coordinated GRL motion"), Proposition 1 ("Residual bound and certificate")
and the numerical results (Section 5) of the paper. Everything is computed from the closed-form expressions
of Ω, Ω′ and Ω″: no grid and no numerical gradient. Tested in MATLAB R2024b; it also runs in GNU Octave.

## Quick start

```matlab
run_certificate_flagship     % Flagship example: N = 5, five circles
run_certificate_warehouse    % Warehouse example: N = 3, seven circles
```

For each reflection depth m = 1, 2, 3 the script prints:

- the critical points found and the number predicted by the argument principle (they must coincide);
- for every circle: residual ε̄ = max|∂Φ/∂n|, oscillation of Ψ, bound π r ε̄ (Prop. 1, item 1), winding of R (must be 0);
- the fraction of certified phases, the best phase and its margin, and the margin of the phase used in the paper (θ = 0).

It also plots the certificate margin as a function of θ (margin > 0: certified phase).
Reference outputs are in `results/r3_out_*.txt` and `results/r3_out_*.png`.

The inputs in `data/` (starts, goals, centres and radii in the canonical circular domain) were obtained with the
conformal map of the PlgCirMap toolbox by M. M. S. Nasser, which is not needed to run the code.

## Monte Carlo comparison (Table 2)

```matlab
R = mc_canonical('flagship',  'data/r3_inputs_flagship.mat',  5, 50, 2, 11);
R = mc_canonical('warehouse', 'data/r3_inputs_warehouse.mat', 5, 50, 2, 11);
mc_summary   % reads mc_flagship.mat and mc_warehouse.mat from the current folder
```

Arguments: domain name, inputs, N robots, number of instances, reflection depth m, random seed.
The saved results of the paper are in `results/` (`mc_*.mat`, `log_*.txt`, `mc_table.txt`).

| Domain | GRL certified / successful | Independent HFM: crossing paths | APF successful |
|---|---|---|---|
| Flagship | 100 % / 100 % | 26 % | 24 % |
| Warehouse | 100 % / 100 % | 16 % | 4 % |

## Files

| File | Content |
|---|---|
| `grl_images.m` | image singularities by iterated Milne–Thomson reflections up to depth m |
| `grl_eval.m`, `grl_omega.m` | Ω, Ω′, Ω″, Ψ in closed form |
| `grl_critical_points.m` | zeros of Ω′ in D (multistart Newton) and check by the argument principle |
| `grl_phase_margins.m` | critical phases Θb, certificate on Θ∂, margin for every θ |
| `grl_trace_branch.m` | branch of GRL(θ) from pole i: exact departure, step in Φ (log-gain), holomorphic Newton correction onto Ω = Φ + j(θ+π); drift of Ψ ≈ 1e-12 |
| `grl_plan.m` | all N branches and the checks of Theorem 1 (arrival, permutation, inside D, separation) |
| `grl_domain.m` | outer/inner circle indices |
| `bl_assign.m` | minimum sum-of-squares assignment (baselines) |
| `bl_hfm.m` | baseline: independent harmonic potential per robot (same images) |
| `bl_apf.m` | baseline: multi-robot artificial potentials (Khatib + inter-robot repulsion) |
| `mc_canonical.m`, `mc_summary.m` | Monte Carlo in the canonical domain and LaTeX table |
| `r3_certificate_check.m` | certificate report (called by the `run_*` scripts) |
| `test_synth.m`, `test_warehouse_like.m` | small synthetic tests |
