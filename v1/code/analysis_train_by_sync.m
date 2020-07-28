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

% Retinal wave data import & preprocessing
async_wavecnt = load('./wavedata/'+async_wave+'/wavecnt.mat'); async_wavecnt = async_wavecnt.wavecnt-1;
async_waveset = cell(2,async_wavecnt); tic % Wave diffuse parameter
for ii = 1:async_wavecnt
    load('./wavedata/'+async_wave+'/wave'+num2str(ii)+'.mat');
    % Diffuse and normalize
    wave_ON = wave_filter(pos_ON,state_ON,sig_wave);
    wave_OFF = wave_filter(pos_OFF,state_OFF,sig_wave);
    async_waveset(1,ii) = {wave_ON};
    async_waveset(2,ii) = {wave_OFF};
end; gpuDevice(1); toc

sync_wavecnt = load('./wavedata/'+sync_wave+'/wavecnt.mat'); sync_wavecnt = sync_wavecnt.wavecnt-1;
sync_waveset = cell(2,sync_wavecnt); tic % Wave diffuse parameter
for ii = 1:sync_wavecnt
    load('./wavedata/'+sync_wave+'/wave'+num2str(ii)+'.mat');
    % Diffuse and normalize
    wave_ON = wave_filter(pos_ON,state_ON,sig_wave);
    wave_OFF = wave_filter(pos_OFF,state_OFF,sig_wave);
    sync_waveset(1,ii) = {wave_ON};
    sync_waveset(2,ii) = {wave_OFF};
end; gpuDevice(1); toc

%% orientation preference
Result = compute_OP(pos_ON,pos_OFF,w0_V1_ON,w0_V1_OFF,w_V1_ON,w_V1_OFF);
op0 = Result(:,1);
op = Result(:,2);

%% Hebbian learning by retinal wave
w_V1_V1 = w0_V1_V1;
rnn_eps = 1e-7; % Learning rate

save(char(folderDir+"/data_weight_matrix/w_sync_epoch0.mat"),'w_V1_V1');
for epoch = 1:10
    w_V1_V1 = V1_Hebbian_update(epoch,sync_wavecnt,sync_waveset,w_V1_ON,w_V1_OFF,w_V1_V1,...
        V1_max,V1_thr,V1_slope,V1_tau,rnn_eps,rnn_w_sum_lim,rnn_w_lim,false);
    gpuDevice(1); % Refresh GPU memory
    save(char(folderDir+"/data_weight_matrix/w_sync_epoch"+num2str(epoch)+".mat"),'w_V1_V1','epoch');
end

% OP-weight trend test
lhc_lim = 500;
figure; analysis_OP_weight_hist(op,w_V1_V1,pos_V1,retina_V1_ratio,lhc_lim);
