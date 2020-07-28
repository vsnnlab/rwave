%% Import data
parallel.gpu.rng(123, 'Philox4x32-10');
addpath('./function_init_model/');
addpath('./function_map_analysis/');

% import w_V1_RGC.mat, w0.mat

%% wave data
wave = "additional_mouse/2020-05-18,14-55";
wave = "additional_monkey/2020-05-19,00-34";
wave = "additional_cat/2020-05-19,20-55";

Result = compute_OP(pos_ON,pos_OFF,w0_V1_ON,w0_V1_OFF,w_V1_ON,w_V1_OFF);
op0 = Result(:,1);
op = Result(:,2);

%% data import & preprocessing
% Import RGC mosaics data from retinal wave dataset
load('./wavedata/'+wave+'/mosaics.mat');
pos_OFF = saved_OFF; pos_ON = saved_ON;
clear crop_OFF crop_ON;
try
    crop_x = crop;
    crop_y = crop;
    disp("legacy parameters adjusted");
catch
    disp("no legacy parameter to adjust");
end

% Retinal wave data import & preprocessing
wavecnt = load('./wavedata/'+wave+'/wavecnt.mat'); wavecnt = wavecnt.wavecnt-1;
waveset = cell(2,wavecnt); sig_wave = d_OFF*0.85; tic % Wave diffuse parameter
wavedir = zeros(wavecnt,1);
crop_win = min(crop_x,crop_y);
ON_center = find(abs(pos_ON(:,1))<crop_win & abs(pos_ON(:,2))<crop_win);
OFF_center = find(abs(pos_OFF(:,1))<crop_win & abs(pos_OFF(:,2))<crop_win);
for ii = 1:wavecnt
    load('./wavedata/'+wave+'/wave'+num2str(ii)+'.mat');
    % Diffuse and normalize
    wave_ON = wave_filter(pos_ON,state_ON,sig_wave);
    wave_OFF = wave_filter(pos_OFF,state_OFF,sig_wave);
    waveset(1,ii) = {wave_ON};
    waveset(2,ii) = {wave_OFF};
    
    % Measure wave direction
    dir_t = round(size(wave_ON,2)/2);
    ON_cm = wave_ON(ON_center,dir_t)'*pos_ON(ON_center,:)/size(ON_center,1);
    OFF_cm = wave_OFF(OFF_center,dir_t)'*pos_OFF(OFF_center,:)/size(OFF_center,1);
    wave_vec = ON_cm-OFF_cm;
    wavedir(ii) = angle(wave_vec(1)+1i*wave_vec(2));
end; gpuDevice(1); toc

% Enforce no bias in directions
dir_categories = -pi:pi/6:pi;
num_per_category = 999;
for ii = 1:size(dir_categories, 2)-1
    inds = find(wavedir >= dir_categories(ii) & wavedir < dir_categories(ii+1));
    if size(inds,1)<num_per_category
        num_per_category = size(inds,1);
    end
end

select_ind = [];
for ii = 1:size(dir_categories, 2)-1
    inds = find(wavedir >= dir_categories(ii) & wavedir < dir_categories(ii+1));
    select_ind = [select_ind; randsample(inds, num_per_category)];
end
wavecnt = size(select_ind,1);
wavedir = wavedir(select_ind);
waveset = waveset(:,select_ind);

figure;
histogram(wavedir,-pi:pi/6:pi);
xticks([-pi 0 pi]);
title("Direction bias");

%% parameters
dt = 0.1; % Simulation time step: 100ms
% Initial RGC-V1 feedforward wiring
if mouse == true
    density_ON = size(pos_ON,1)/4/crop_x/crop_y;
    density_OFF = size(pos_OFF,1)/4/crop_x/crop_y;
    density_V1 = (density_ON+density_OFF)*0.1;
    d_V1 = sqrt(2/sqrt(3)/density_V1);
else
    d_V1 = 0;
end

% V1 response curve
V1_thr = 0.5;
V1_slope = 0.15;
V1_max = 1;

% Horizontal connection learning
rnn_r_lim = 360;
rnn_eps = 2.5e-5; % Learning rate, mouse
rnn_eps = 1e-8; % Learning rate, monkey
rnn_eps = 2e-7; % Learning rate, cat
rnn_w_sum_lim = 0.01; rnn_w_lim = rnn_w_sum_lim/20; % Resource limit

%% Analysis
V1_tau_default = 15; % Relative size of response averaging window to inter-wave period
% Assume V1_tau represents a constant time period
% V1_tau small: less wave per time period
% V1_tau large: more wave per time period

relative_frequency = (10:2:20) * 0.1;
num_repeat = 5;
V1_tau_list = V1_tau_default*relative_frequency;
iter_all = zeros(num_repeat, 6);
for ii = 1:length(V1_tau_list)
    V1_tau = V1_tau_list(ii);
    
    for n = 1:num_repeat
        w_V1_V1 = w0_V1_V1;
        iteration = 0;
        
        while (1)
            iteration = iteration + 1;
            % Constants
            n_ON = size(w_V1_ON,2);
            n_OFF = size(w_V1_OFF,2);
            n_V1 = size(w_V1_V1,1);
            alpha = 1/V1_tau;
            % Buffer
            V1_rate = zeros(n_V1,1);
            V1_running_avg = zeros(n_V1,1);
            
            wave_num = randi(wavecnt, 1);
            
            % zero padding
            wave_ON = [zeros(n_ON,10) cell2mat(waveset(1,wave_num)) zeros(n_ON,10)];
            wave_OFF = [zeros(n_OFF,10) cell2mat(waveset(2,wave_num)) zeros(n_OFF,10)];
            time = size(wave_OFF,2);
            
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
            
            % OP-weight trend test
            lhc_lim = 0;
            figure; p = analysis_OP_weight_hist(op,w_V1_V1,pos_V1,retina_V1_ratio,lhc_lim); close;
            if (p < 0.05)
                break;
            end
            if (iteration > 50)
                w_V1_V1 = w0_V1_V1;
                iteration = 0;
            end
        end
        
        iter_all(n, ii) = iteration;
    end
end

p = anova1(iter_all./reshape(V1_tau_list, [1, 6]));
iter_mean = mean(iter_all, 1);
iter_std = std(iter_all, 1);

figure; errorbar(relative_frequency*V1_tau_default, iter_mean./V1_tau_list, iter_std./V1_tau_list);
xlabel('Relative wave frequency, (waves/tau)');
ylabel('Taken time to be ori-specific (tau)');
title("ANOVA p = " + num2str(p));
xlim([14 31]); ylim([0 max(iter_mean./V1_tau_list + iter_std./V1_tau_list) + 0.1]);
xticks(relative_frequency*V1_tau_default);
xticklabels(relative_frequency*V1_tau_default);
