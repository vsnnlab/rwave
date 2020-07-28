function [Result] = ff_Hebbian_update(epoch,wavecnt,waveset,w_V1_ON,w_V1_OFF,V1_max,V1_thr,V1_slope,V1_tau,ff_eps,ff_w_sum_lim,ff_w_lim)
tic;
% Constants
n_ON = size(w_V1_ON,2);
n_OFF = size(w_V1_OFF,2);
n_V1 = size(w_V1_ON,1);
alpha = 1/V1_tau;
% Buffers
ON_running_avg = zeros(n_ON,n_V1);
OFF_running_avg = zeros(n_OFF,n_V1);
V1_running_avg = zeros(n_V1,1);

for wave_num = randperm(wavecnt)
fprintf("Updating ff wiring, using wave #%d/#%d... epoch %d\n",wave_num,wavecnt,epoch);
% zero padding
wave_ON = [zeros(n_ON,10) cell2mat(waveset(1,wave_num)) zeros(n_ON,10)];
wave_OFF = [zeros(n_OFF,10) cell2mat(waveset(2,wave_num)) zeros(n_OFF,10)];

wave_ON = gpuArray(wave_ON);
wave_OFF = gpuArray(wave_OFF); % RGC X TIME (padded)
w_V1_ON = gpuArray(w_V1_ON);
w_V1_OFF = gpuArray(w_V1_OFF); % V1 X RGC

% V1 responses
V1_input = w_V1_ON*wave_ON + w_V1_OFF*wave_OFF; % V1 X TIME
V1_rate = V1_max*logsig((V1_input-V1_thr)/V1_slope); % V1 X TIME

% For each V1 neuron, only consider RGC response profiles that maximize V1 response
[V1_max_rate, extract] = max(V1_rate,[],2); % time index to extract
extract_V1 = V1_max_rate;
extract_ON = wave_ON(:,extract);
extract_OFF = wave_OFF(:,extract);

% Running averages of extracted reponse profiles
V1_running_avg = extract_V1*alpha + V1_running_avg*(1-alpha);
ON_running_avg = extract_ON*alpha + ON_running_avg*(1-alpha);
OFF_running_avg = extract_OFF*alpha + OFF_running_avg*(1-alpha);

% Computing covariance
V1_del = extract_V1-V1_running_avg;
V1_del = reshape(V1_del,[1 n_V1]);
ON_del = extract_ON-ON_running_avg;
OFF_del = extract_OFF-OFF_running_avg;
covr_V1_ON = (ON_del.*V1_del)';
covr_V1_OFF = (OFF_del.*V1_del)';

% Covariance rule: dw_ij/dt = eps*(ri-avg(ri))*(rj-(avg(rj)))
% https://www.sciencedirect.com/science/article/pii/B9780121489557500102
del_w_V1_ON = ff_eps*covr_V1_ON.*(w_V1_ON>0);
del_w_V1_OFF = ff_eps*covr_V1_OFF.*(w_V1_OFF>0);

% Weight update
new_V1_ON = w_V1_ON + del_w_V1_ON;
new_V1_OFF = w_V1_OFF + del_w_V1_OFF;

% Input limit & nomalization
input_sum = sum(new_V1_ON,2) + sum(new_V1_OFF,2);
bounded = min(input_sum,ff_w_sum_lim);
new_V1_ON = new_V1_ON./input_sum.*bounded;
new_V1_OFF = new_V1_OFF./input_sum.*bounded;
new_V1_ON(isnan(new_V1_ON)) = 0;
new_V1_OFF(isnan(new_V1_OFF)) = 0;
% No negative weights
new_V1_ON(new_V1_ON<0) = 0;
new_V1_OFF(new_V1_OFF<0) = 0;
% Individual weight limit
new_V1_ON(new_V1_ON>ff_w_lim) = ff_w_lim;
new_V1_OFF(new_V1_OFF>ff_w_lim) = ff_w_lim;
w_V1_ON = new_V1_ON;
w_V1_OFF = new_V1_OFF;
end

% Retrieve
Result = {gather(w_V1_ON),gather(w_V1_OFF)};
gpuDevice(1); toc
end