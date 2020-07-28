function [Result] = async_propagate(step,state_ON,state_AC,state_OFF,w_ON_ON,w_AC_ON,w_OFF_AC,thr_ON,thr_AC,thr_OFF,steps_fire)
% Cellular automata: Current state is based on state just before
state_ON_past = gpuArray(state_ON(:,step-1));
ON_firing_past = double(state_ON_past == 2);

% Load arrays on GPU
w_ON_ON_gpu = gpuArray(w_ON_ON);
w_AC_ON_gpu = gpuArray(w_AC_ON);
w_OFF_AC_gpu = gpuArray(w_OFF_AC);

% ON RGCs receive input from other ON RGCs
w_ON_ON_rand = w_ON_ON_gpu.*(gpuArray.randn(size(w_ON_ON_gpu))*0.2+1); % randomly varying cell coupling
overthr_ON = (w_ON_ON_rand*ON_firing_past)>thr_ON; % Cells receiving over-threshold input
recruitable_ON = state_ON_past == 0; % Cells currently recruitable
firing_ON = overthr_ON&recruitable_ON; % Cells that are going to fire
state_ON(firing_ON,step:step+steps_fire-1) = 2; % Fire
state_ON(firing_ON,step+steps_fire:end) = 1; % Refractory period

% ACs receive input from ON RGCs
% Immediate transduction, assumed to have no refractory period
w_AC_ON_rand = w_AC_ON_gpu.*(gpuArray.randn(size(w_AC_ON_gpu))*0.2+1); % randomly varying cell coupling
firing_AC = (w_AC_ON_rand*ON_firing_past)>thr_AC; % Cells receiving over-threshold input
state_AC(firing_AC,step) = 1; % Depolarize
state_AC(~firing_AC,step) = 0; % Wait

% OFF RGCs receive inhibitory input from ACs
w_OFF_AC_rand = w_OFF_AC_gpu.*(gpuArray.randn(size(w_OFF_AC_gpu))*0.2+1); % randomly varying cell coupling
input_OFF = w_OFF_AC_rand*firing_AC;

% Hyperpolarize
underthr_OFF = input_OFF<thr_OFF; % Cells receiving over-threshold inhibition
recruitable_OFF = state_OFF(:,step-1) == 0; % Cells currently recruitable
hyperpolarize_OFF = underthr_OFF&recruitable_OFF; % Cells that are going to be hyperpolarized
state_OFF(hyperpolarize_OFF,step) = -1; % Hyperpolarize

% Fire
OFF_hyperpolarize_past = state_OFF(:,step-1) == -1; % Hypolarized cells
overthr_OFF = input_OFF>=thr_OFF; % Cells receiving over-threshold input
firing_OFF = overthr_OFF&OFF_hyperpolarize_past; % Cells that are going to fire
state_OFF(firing_OFF,step:step+steps_fire-1) = 2; % Fire
state_OFF(firing_OFF,step+steps_fire:end) = 1; % Refractory period
state_OFF((~overthr_OFF)&OFF_hyperpolarize_past,step) = -1; % Remain hypolarized
Result = {gather(state_ON),gather(state_AC),gather(state_OFF)};
end