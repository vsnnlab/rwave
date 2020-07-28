%% Import data
parallel.gpu.rng(123, 'Philox4x32-10');
addpath('./function_init_model/');
addpath('./function_map_analysis/');

% import w_V1_RGC.mat

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
sig_wave = d_OFF*0.85;
IMGSIZE = 100;

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

imgsize_x = 200; % Filtered map width
img_sig = 9; % Gaussian image filtering width (unit: pixels)
img_sig = 7; % Gaussian image filtering width (unit: pixels)
% V1 response curve
V1_thr = 0.5;
V1_slope = 0.15;
V1_max = 1;
V1_tau = 15; % Response averaging window


%% Pairwise correlation analysis
% (V1, ON, OFF) triplets
[OFF_maxw, OFF_maxw_idx] = max(w_V1_OFF,[],2);
[ON_maxw, ON_maxw_idx] = max(w_V1_ON,[],2);

% signal crop size: 2*crop_wsize
crop_wsize = 10;

% Async waves
% compute V1 response
async_r = [];
for waveidx = 1:100
    wave_ON = cell2mat(async_waveset(1, waveidx));
    wave_OFF = cell2mat(async_waveset(2, waveidx));
    V1_rate = V1_max*logsig(((w_V1_ON*wave_ON+w_V1_OFF*wave_OFF)-V1_thr)/V1_slope); % V1 X TIME
    [~, maxt] = max(V1_rate, [], 2);
    
    for V1_idx = 1:size(w_V1_OFF, 1)
        % crop window: 20 steps around V1 peak
        t_range = max(maxt(V1_idx) - crop_wsize, 1) : min(maxt(V1_idx) + crop_wsize, size(V1_rate, 2));
        OFF_idx = OFF_maxw_idx(V1_idx);
        ON_idx = ON_maxw_idx(V1_idx);
        OFF_V1_R = corrcoef(wave_OFF(OFF_idx, t_range), V1_rate(V1_idx, t_range));
        ON_V1_R = corrcoef(wave_ON(ON_idx, t_range), V1_rate(V1_idx, t_range));
        
%         figure; hold on;
%         plot(t_range, wave_OFF(OFF_idx, t_range));
%         plot(t_range, wave_ON(ON_idx, t_range));
%         plot(t_range, V1_rate(V1_idx, t_range));
        [OFF_V1_R(2,1), ON_V1_R(2,1)]
        async_r = [async_r, OFF_V1_R(2,1), ON_V1_R(2,1)];
    end
end

% Sync waves
% compute V1 response
sync_r = [];
for waveidx = 1:100
    wave_ON = cell2mat(sync_waveset(1, waveidx));
    wave_OFF = cell2mat(sync_waveset(2, waveidx));
    V1_rate = V1_max*logsig(((w_V1_ON*wave_ON+w_V1_OFF*wave_OFF)-V1_thr)/V1_slope); % V1 X TIME
    [~, maxt] = max(V1_rate, [], 2);
    
    for V1_idx = 1:size(w_V1_OFF, 1)
        % crop window: 20 steps around V1 peak
        t_range = max(maxt(V1_idx) - crop_wsize, 1) : min(maxt(V1_idx) + crop_wsize, size(V1_rate, 2));
        OFF_idx = OFF_maxw_idx(V1_idx);
        ON_idx = ON_maxw_idx(V1_idx);
        OFF_V1_R = corrcoef(wave_OFF(OFF_idx, t_range), V1_rate(V1_idx, t_range));
        ON_V1_R = corrcoef(wave_ON(ON_idx, t_range), V1_rate(V1_idx, t_range));
        
%         figure; hold on;
%         plot(t_range, wave_OFF(OFF_idx, t_range));
%         plot(t_range, wave_ON(ON_idx, t_range));
%         plot(t_range, V1_rate(V1_idx, t_range));

        [OFF_V1_R(2,1), ON_V1_R(2,1)]
        sync_r = [sync_r, OFF_V1_R(2,1), ON_V1_R(2,1)];
    end
end

[h, p] = ttest2(async_r, sync_r);
figure;
errorbar([1,2],[nanmean(sync_r),nanmean(async_r)], [nanstd(sync_r),nanstd(async_r)]);
xlim([0.5 2.5]); ylim([-0.2 1.2]);
suptitle("RGC-V1 pairwise. sync: " + nanmean(sync_r) + "+-" + nanstd(sync_r) + ...
    " / async: " + nanmean(async_r) + "+-" + nanstd(async_r) + ...
    ", p = " + num2str(p) + ", n = m = " + num2str(100*size(w_V1_OFF, 1)));

figure; hold on;
plot([1, 1], [prctile(sync_r, 25), prctile(sync_r, 75)], '-k');
plot([2, 2], [prctile(async_r, 25), prctile(async_r, 75)], '-k');
plot([1, 2], [nanmean(sync_r), nanmean(async_r)], '-k');
xlim([0.5 2.5]); ylim([0 1]);
suptitle("RGC-V1 pairwise. sync: [" + prctile(sync_r, 25) + "," + prctile(sync_r, 75) + ...
    "] / async: [" +  + prctile(async_r, 25) + "," + prctile(async_r, 75) + ...
    "], p = " + num2str(p) + ", n = m = " + num2str(100*size(w_V1_OFF, 1)));

%% Full-field correlation analysis
pos_RGC = [pos_OFF; pos_ON];
crop_idx = find(abs(pos_RGC(:,1))<=crop_x&abs(pos_RGC(:,2))<=crop_y);

imgsize_x = 10;
img_sig = 1;

% Async waves
% compute retina/V1 activity images
async_r = [];
for waveidx = 1:100
    wave_ON = cell2mat(async_waveset(1, waveidx));
    wave_OFF = cell2mat(async_waveset(2, waveidx));
    V1_rate = V1_max*logsig(((w_V1_ON*wave_ON+w_V1_OFF*wave_OFF)-V1_thr)/V1_slope); % V1 X TIME
    
    t_img = round(size(V1_rate, 2)/2);
    wave_RGC = [wave_ON; wave_OFF];
    RGC_img = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_RGC(crop_idx,:),wave_RGC(crop_idx, t_img),true);
    V1_img = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_V1,V1_rate(:, t_img),true);
    
%     figure; colormap jet;
%     subplot(221); imagesc(RGC_img); axis image xy; colorbar; caxis([0 1]);
%     subplot(222); imagesc(V1_img); axis image xy; colorbar; caxis([0 1]);
%     subplot(223); scatter(pos_RGC(crop_idx,1),pos_RGC(crop_idx,2),30,wave_RGC(crop_idx, t_img),'filled','markeredgecolor','r');
%     axis image xy; colorbar; caxis([0 1]);
%     subplot(224); scatter(pos_V1(:,1),pos_V1(:,2),30, V1_rate(:, t_img),'filled','markeredgecolor','r');
%     axis image xy; colorbar; caxis([0 1]);
    
    img_corr = corrcoef(RGC_img(:), V1_img(:));
    img_corr(2, 1)
    async_r = [async_r, img_corr(2, 1)];
end

% Sync waves
% compute retina/V1 activity images
sync_r = [];
for waveidx = 1:100
    wave_ON = cell2mat(sync_waveset(1, waveidx));
    wave_OFF = cell2mat(sync_waveset(2, waveidx));
    V1_rate = V1_max*logsig(((w_V1_ON*wave_ON+w_V1_OFF*wave_OFF)-V1_thr)/V1_slope); % V1 X TIME
    
    t_img = randi([1 size(V1_rate, 2)],1);
    wave_RGC = [wave_ON; wave_OFF];
    RGC_img = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_RGC(crop_idx,:),wave_RGC(crop_idx, t_img),true);
    V1_img = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_V1,V1_rate(:, t_img),true);
    
%     figure; colormap jet;
%     subplot(221); imagesc(RGC_img); axis image xy; colorbar; caxis([0 1]);
%     subplot(222); imagesc(V1_img); axis image xy; colorbar; caxis([0 1]);
%     subplot(223); scatter(pos_RGC(crop_idx,1),pos_RGC(crop_idx,2),30,wave_RGC(crop_idx, t_img),'filled','markeredgecolor','r');
%     axis image xy; colorbar; caxis([0 1]);
%     subplot(224); scatter(pos_V1(:,1),pos_V1(:,2),30, V1_rate(:, t_img),'filled','markeredgecolor','r');
%     axis image xy; colorbar; caxis([0 1]);
    
    img_corr = corrcoef(RGC_img(:), V1_img(:));
    img_corr(2, 1)
    sync_r = [sync_r, img_corr(2, 1)];
end

[h, p] = ttest2(async_r, sync_r);
figure;
errorbar([1,2],[nanmean(sync_r),nanmean(async_r)], [nanstd(sync_r),nanstd(async_r)]);
xlim([0.5 2.5]);% ylim([-0.2 1.2]);
suptitle("RGC-V1 global. sync: " + nanmean(sync_r) + "+-" + nanstd(sync_r) + ...
    " / async: " + nanmean(async_r) + "+-" + nanstd(async_r) + ...
    ", p = " + num2str(p) + "n = m = " + num2str(100*size(w_V1_OFF, 1)));
