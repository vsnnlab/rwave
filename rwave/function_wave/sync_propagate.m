function [Result] = sync_propagate(step,state_ON,state_OFF,w_ON_ON,w_OFF_ON,thr_ON,thr_OFF,steps_fire)
% Cellular automata: Current state is based on state just before
state_ON_past = gpuArray(state_ON(:,step-1));
ON_firing_past = double(state_ON_past == 2);
state_OFF_past = gpuArray(state_OFF(:,step-1));

% Load arrays on GPU
w_ON_ON_gpu = gpuArray(w_ON_ON);
w_OFF_ON_gpu = gpuArray(w_OFF_ON);

% ON RGCs receive input from other ON RGCs
w_ON_ON_rand = w_ON_ON_gpu.*(gpuArray.randn(size(w_ON_ON_gpu))*0.2+1); % randomly varying cell coupling
overthr_ON = (w_ON_ON_rand*ON_firing_past)>thr_ON; % Cells receiving over-threshold input
recruitable_ON = state_ON_past == 0; % Cells currently recruitable
firing_ON = overthr_ON&recruitable_ON; % Cells that are going to fire
state_ON(firing_ON,step:step+steps_fire-1) = 2; % Fire
state_ON(firing_ON,step+steps_fire:end) = 1; % Refractory period

% OFFs receive input from ON RGCs
w_OFF_ON_rand = w_OFF_ON_gpu.*(gpuArray.randn(size(w_OFF_ON_gpu))*0.2+1); % randomly varying cell coupling
overthr_OFF = (w_OFF_ON_rand*ON_firing_past)>thr_OFF; % Cells receiving over-threshold input
recruitable_OFF = state_OFF_past == 0; % Cells currently recruitable
firing_OFF = overthr_OFF&recruitable_OFF; % Cells that are going to fire
state_OFF(firing_OFF,step:step+steps_fire-1) = 2; % Fire
state_OFF(firing_OFF,step+steps_fire:end) = 1; % Refractory period

Result = {gather(state_ON),gather(state_OFF)};
end