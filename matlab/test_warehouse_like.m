% symmetric test resembling the warehouse canonical domain (circles on real axis)
centers = [-0.9078; -0.6818; -0.2654; 0.2654; 0.6818; 0.9078; 0];
radii   = [0.06222; 0.1272; 0.2164; 0.2164; 0.1272; 0.06222; 1];
rob_i_z = [-0.80+0.45i, -0.75-0.45i, 0.82+0.02i];
rob_g_z = [-0.47+0.02i, 0.0+0.05i, 0.47-0.03i];
N = 3;
for m = 1:2
  [P, Z] = grl_images(rob_i_z, rob_g_z, centers, radii, m);
  tic; [b, nexp, nf] = grl_critical_points(P, Z, N, centers, radii); 
  fprintf('m=%d found %d expected %d (%.1fs)\n', m, nf, nexp, toc);
end
