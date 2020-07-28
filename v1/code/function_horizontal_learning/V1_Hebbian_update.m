function [Result] = V1_Hebbian_update...
    (epoch,wavecnt,waveset,w_V1_ON,w_V1_OFF,w_V1_V1,V1_max,V1_thr,V1_slope,V1_tau,...
    rnn_eps,rnn_w_sum_lim,rnn_w_lim,shuffle)
tic;
w_V1_ON = gpuArray(w_V1_ON);
w_V1_OFF = gpuArray(w_V1_OFF); % V1 X RGC
w_V1_V1 = gpuArray(w_V1_V1);

% Constants
n_ON = size(w_V1_ON,2);
n_OFF = size(w_V1_OFF,2);
n_V1 = size(w_V1_V1,1);
alpha = 1/V1_tau;
% Buffer
V1_rate = zeros(n_V1,1);
V1_running_avg = zeros(n_V1,1);

for wave_num = randperm(wavecnt)
fprintf("Updating cortical wiring, using wave #%d/#%d... epoch %d\n",wave_num,wavecnt,epoch);
% zero padding
wave_ON = [zeros(n_ON,10) cell2mat(waveset(1,wave_num)) zeros(n_ON,10)];
wave_OFF = [zeros(n_OFF,10) cell2mat(waveset(2,wave_num)) zeros(n_OFF,10)];
time = size(wave_OFF,2);

% Wave input
if shuffle
    wave_ON = wave_ON(randperm(n_ON), :);
    wave_OFF = wave_OFF(randperm(n_OFF), :);
end

wave_ON = gpuArray(wave_ON);
wave_OFF = gpuArray(wave_OFF); % RGC X TIME (padded)
V1_ff_input = w_V1_ON*wave_ON + w_V1_OFF*wave_OFF; % V1 X TIME

% Max rate buffer
V1_maxrate = zeros(n_V1,1);

% Compute rate(t) and max(rate(t))
for tt = 1:size(V1_ff_input,2)
V1_rec_input = w_V1_V1*V1_rate; % V1 X TIME
V1_input = V1_ff_input(:,tt) + V1_rec_input;
V1_rate = V1_max*logsig((V1_input-V1_thr)/V1_slope); % V1 X TIME
V1_maxrate = max(V1_rate,V1_maxrate);
end

% Running average of max rate
V1_running_avg = V1_maxrate*alpha + V1_running_avg*(1-alpha);

% Computing covariance
V1_del = V1_maxrate-V1_running_avg;
covr = V1_del.*V1_del';

% Covariance rule: dw_ij/dt = eps*(ri-avg(ri))*(rj-(avg(rj)))
% https://www.sciencedirect.com/science/article/pii/B9780121489557500102
del_w = rnn_eps*covr.*(w_V1_V1>0);

% Weight update
w_V1_V1 = w_V1_V1 + del_w;

% No autosynapse
w_V1_V1 = w_V1_V1-diag(diag(w_V1_V1));
% Input limit & nomalization
input_sum = sum(w_V1_V1,2);
bounded = min(input_sum,rnn_w_sum_lim);
w_V1_V1 = w_V1_V1./input_sum.*bounded;

output_sum = sum(w_V1_V1,1);
bounded = min(output_sum,rnn_w_sum_lim);
w_V1_V1 = w_V1_V1./output_sum.*bounded;

w_V1_V1(isnan(w_V1_V1)) = 0;
% No negative weights
w_V1_V1(w_V1_V1<0) = 0;
% Individual weight limit
w_V1_V1(w_V1_V1>rnn_w_lim) = rnn_w_lim;
end

% Retrieve
Result = gather(w_V1_V1);
gpuDevice(1); toc
end