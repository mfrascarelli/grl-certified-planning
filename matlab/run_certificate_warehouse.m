%RUN_CERTIFICATE_WAREHOUSE  Certificate check on the Warehouse example (N = 3, seven circles).
%   Starts, goals and circles of the canonical domain D are loaded from data/.
%   They were obtained with the conformal map of the PlgCirMap toolbox (Nasser),
%   which is not needed to run this script.
clear; close all; clc;
here = fileparts(mfilename('fullpath')); addpath(here);
load(fullfile(here, 'data', 'r3_inputs_warehouse.mat'));   % rob_i_z, rob_g_z, centers, radii
n_inv = 3;          % reflection depths m = 1..3
theta_used = 0;     % phase used in the paper figures
r3_certificate_check
