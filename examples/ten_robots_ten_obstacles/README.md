# Ten robots, ten circular obstacles (unit disk)

Certified GRL planning (Algorithm 1) with iterated Milne–Thomson reflections, depth m = 3.

    python pipeline.py 3 3      # potential, critical points (+ argument principle), Theta_b, certificate, theta*   (~5 min)
    python branches.py 3 3      # the N branches of GRL(theta*), assignment, clearances                            (~5 s)
    python figures.py           # Forest_Omega_obs.eps, Forest_solution.eps, Forest_phases.eps                     (~1.5 min)

Arguments: random seed of the obstacle layout, reflection depth. `grl_core.py` is the numerical core of the companion notebook.

The certificate uses the second-order sampling bound
|Psi(w) - Psi(z_l)| <= (h/2)|Omega'(z_l)| + h^2 K_i / 8,  K_i = sum_xi dist(xi, C_i)^(-2),
because with 24 420 image singularities the first-order constant L_i h_i / 2 would require millions of samples per circle.
