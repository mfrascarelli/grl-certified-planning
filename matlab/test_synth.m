rand('seed',3);
centers = [0; 0.45+0.1i; -0.35-0.3i; -0.2+0.5i]; radii = [1; 0.18; 0.15; 0.12];
N = 4; rob_i_z = []; rob_g_z = [];
while numel(rob_i_z) < N
  z = (rand-0.5)*1.8 + 1i*(rand-0.5)*1.8;
  if abs(z)<0.9 && all(abs(z-centers(2:end))>radii(2:end)+0.05), rob_i_z(end+1)=z; end
end
while numel(rob_g_z) < N
  z = (rand-0.5)*1.8 + 1i*(rand-0.5)*1.8;
  if abs(z)<0.9 && all(abs(z-centers(2:end))>radii(2:end)+0.05), rob_g_z(end+1)=z; end
end
n_inv = 3; theta_used = 0;
r3_certificate_check
