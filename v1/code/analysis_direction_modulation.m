%% Import data
parallel.gpu.rng(123, 'Philox4x32-10');
addpath('./function_init_model/');
addpath('./function_map_analysis/');

% import w_V1_RGC.mat, w0.mat

% wave data
sync_wave = "additional_mouse/2020-05-18,14-53";
async_wave = "additional_mouse/2020-05-18,14-55";
sync_wave = "additional_monkey/2020-05-19,08-03";
async_wave = "additional_monkey/2020-05-19,00-34";
sync_wave = "additional_cat/2020-05-19,21-46";
async_wave = "additional_cat/2020-05-19,20-55";

% Import RGC mosaics data from retinal wave dataset
load('./wavedata/'+async_wave+'/mosaics.mat');
pos_OFF = saved_OFF; pos_ON = saved_ON;
clear crop_OFF crop_ON;
try
    crop_x = crop;
    crop_y = crop;
    disp("legacy parameters adjusted");
catch
    disp("no legacy parameter to adjust");
end

% orientation preference
Result = compute_OP(pos_ON,pos_OFF,w0_V1_ON,w0_V1_OFF,w_V1_ON,w_V1_OFF);
op0 = Result(:,1);
op = Result(:,2);

%% unbiased data import & preprocessing
wavecnt = load('./wavedata/'+async_wave+'/wavecnt.mat');
wavecnt = wavecnt.wavecnt-1;
waveset = cell(2,wavecnt); tic % Wave diffuse parameter
for ii = 1:wavecnt
    load('./wavedata/'+async_wave+'/wave'+num2str(ii)+'.mat');
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
wavecnt = size(select_ind,1);
wavedir = wavedir(select_ind);
waveset = waveset(:,select_ind);

figure;
histogram(wavedir,-pi:pi/6:pi);
xticks([-pi 0 pi]);
title("Direction bias");

%% unbiased control
w_V1_V1 = w0_V1_V1;
rnn_eps = 1e-4; % Learning rate

save(char(folderDir+"/data_weight_matrix/w_sync_epoch0.mat"),'w_V1_V1');
for epoch = 1:2
    w_V1_V1 = V1_Hebbian_update(epoch,wavecnt,waveset,w_V1_ON,w_V1_OFF,w_V1_V1,...
        V1_max,V1_thr,V1_slope,V1_tau,rnn_eps,rnn_w_sum_lim,rnn_w_lim,false);
    gpuDevice(1); % Refresh GPU memory
    save(char(folderDir+"/data_weight_matrix/w_unbias_epoch"+num2str(epoch)+".mat"),'w_V1_V1','epoch');
end

% OP-weight trend test
figure; analysis_OP_weight_hist(op,w_V1_V1,pos_V1,retina_V1_ratio,lhc_lim);

%% Main loop
num_bias = length(dir_categories) - 1;
learned_biased = zeros(2, num_bias);
learned_unbiased = zeros(2, num_bias);
rnn_eps = 1e-2; % Learning rate

for nn = 1:num_bias
    % biased data import & preprocessing
    biased_idx = nn;
    select_ind = [];
    for ii = 1:size(dir_categories, 2)-1
        inds = find(wavedir >= dir_categories(ii) & wavedir < dir_categories(ii+1));
        if ii == biased_idx
            select_ind = [select_ind; inds];
        end
    end
    bias_wavecnt = size(select_ind,1);
    bias_wavedir = wavedir(select_ind);
    bias_waveset = waveset(:,select_ind);
    
    % Hebbian learning by retinal wave
    w_biased_V1_V1 = w0_V1_V1;
    
    save(char(folderDir+"/data_weight_matrix/w_sync_epoch0.mat"),'w_V1_V1');
    for epoch = 1:10
        w_biased_V1_V1 = V1_Hebbian_update(epoch,bias_wavecnt,bias_waveset,w_V1_ON,w_V1_OFF,w_biased_V1_V1,...
            V1_max,V1_thr,V1_slope,V1_tau,rnn_eps,rnn_w_sum_lim,rnn_w_lim,false);
    end
    gpuDevice(1); % Refresh GPU memory
    
    % Biased specificity analysis
    biased_dir = (dir_categories(biased_idx) + dir_categories(biased_idx+1)) / 2;
    biased_ori = biased_dir + pi/2;
    orth_ori = biased_ori + pi/2;
    if biased_ori > pi/2; biased_ori = biased_ori - pi; end
    if biased_ori < -pi/2; biased_ori = biased_ori + pi; end
    if orth_ori > pi/2; orth_ori = orth_ori - pi; end
    if orth_ori < -pi/2; orth_ori = orth_ori + pi; end
    
    bias_mask = abs(op - biased_ori) < pi/12;
    w_bias_mask = (bias_mask + bias_mask') == 2;
    
    orth_mask = abs(op - orth_ori) < pi/12;
    w_orth_mask = (orth_mask + orth_mask') == 2;
    
    % trained by biased waves
    biased_w = w_biased_V1_V1.*w_bias_mask;
    biased_w = biased_w(:);
    biased_w = biased_w(biased_w>1e-7);
    orth_w = w_biased_V1_V1.*w_orth_mask;
    orth_w = orth_w(:);
    orth_w = orth_w(orth_w>1e-7);
    
    learned_biased(1, nn) = mean(biased_w);
    learned_biased(2, nn) = mean(orth_w);
    
    % trained by unbiased waves
    biased_w = w_V1_V1.*w_bias_mask;
    biased_w = biased_w(:);
    biased_w = biased_w(biased_w>1e-7);
    orth_w = w_V1_V1.*w_orth_mask;
    orth_w = orth_w(:);
    orth_w = orth_w(orth_w>1e-7);
    
    learned_unbiased(1, nn) = mean(biased_w);
    learned_unbiased(2, nn) = mean(orth_w);

end

%% test
biased_w = learned_biased(1, :);
orth_w = learned_biased(2, :);
[h, p] = ttest(biased_w, orth_w);
    figure;
    errorbar([1,2],[nanmean(biased_w),nanmean(orth_w)], [nanstd(biased_w),nanstd(orth_w)]);
    xlim([0.5 2.5]);
    suptitle("<biased> biased w: " + num2str(nanmean(biased_w)) + "+-" + num2str(nanstd(biased_w)) + ...
        " / orthogonal w: " + num2str(nanmean(orth_w)) + "+-" + num2str(nanstd(orth_w)) + ...
        ", p = " + num2str(p) + ", n = m = " + num2str(length(orth_w)));

    
biased_w = learned_unbiased(1, :);
orth_w = learned_unbiased(2, :);
[h, p] = ttest(biased_w, orth_w);
    figure;
    errorbar([1,2],[nanmean(biased_w),nanmean(orth_w)], [nanstd(biased_w),nanstd(orth_w)]);
    xlim([0.5 2.5]);
    suptitle("<unbiased> biased w: " + num2str(nanmean(biased_w)) + "+-" + num2str(nanstd(biased_w)) + ...
        " / orthogonal w: " + num2str(nanmean(orth_w)) + "+-" + num2str(nanstd(orth_w)) + ...
        ", p = " + num2str(p) + ", n = m = " + num2str(length(orth_w)));
