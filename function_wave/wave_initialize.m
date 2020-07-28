function [Result] = wave_initialize(pos_ON,pos_OFF,pos_AC,f,steps,steps_fire,start_r,start_size)
% Cell states
state_ON = gpuArray(zeros(size(pos_ON,1),steps)); % time X ON, Recruitable (0) / Refractory (1) / Firing (2)
state_OFF = gpuArray(zeros(size(pos_OFF,1),steps)); % time X OFF, Recruitable (0) / Inhibited (-1) / Refractory (1) / Firing (2)
state_AC = gpuArray(zeros(size(pos_AC,1),steps)); % time X AC, Recruitable (0) / Active (1)
% Assign random ON cells as "refractory" and others as "recruitable"
% Refractory cells stay refractory during the wave
ON_refractory = gpuArray(rand(size(pos_ON,1),1)>=f);
state_ON(:,:) = double(repmat(ON_refractory,1,steps));
% Declare initiation point
start_theta = rand*2*pi;
[start_x, start_y] = pol2cart(start_theta,start_r);
% Set initially firing ON RGCs
dist2 = gpuArray(sum((pos_ON-[start_x start_y]).^2,2));
ON_near = dist2<=start_size^2;
ON_recruitable = ~ON_refractory;
ON_firing = ON_near&ON_recruitable;
state_ON(ON_firing,1:steps_fire) = 2;
state_ON(ON_firing,steps_fire+1:end) = 1;
Result = {gather(state_ON),gather(state_AC),gather(state_OFF)};
end