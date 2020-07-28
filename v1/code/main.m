clear all; close all; % close(myVideo);
parallel.gpu.rng(123, 'Philox4x32-10');

% Export directory
folderDir = "./export/" + datestr(now,'yyyy-mm-dd,HH-MM') + "_cat_async";
mkdir(folderDir);
mkdir(folderDir+'/data_weight_matrix');
mkdir(folderDir+'/figure_learning');
mkdir(folderDir+'/figure_spontaneous_activity');

% wavedata for main analysis
%WaveDataset = "2019-09-18,16-33"; % Cat, Zhan et al. (2000)
%WaveDataset = "2020-02-01,19-36"; % Monkey, Gauthier et al. (2009)
%WaveDataset = "2020-02-22,13-22"; % Mouse, Bleckert et al. (2014)

% wavedata for additional analysis
%WaveDataset = "additional_mouse/2020-05-18,14-53" % Mouse sync
%WaveDataset = "additional_mouse/2020-05-18,14-55"; % Mouse async
%WaveDataset = "additional_cat/2020-05-19,21-46" % Cat sync
WaveDataset = "additional_cat/2020-05-19,20-55"; % Cat async
%WaveDataset = "additional_monkey/2020-05-19,00-32" % Monkey sync
%WaveDataset = "additional_monkey/2020-05-19,00-34"; % Monkey async

mouse = false;
retina_V1_ratio = 0.75; % cat
%retina_V1_ratio = 1.98; % monkey
%retina_V1_ratio = 0.2; % mouse

% Paths
addpath('./function_horizontal_learning/');
addpath('./function_init_model/');
addpath('./function_map_analysis/');
addpath('./function_utils/');
addpath('./wavedata/'+WaveDataset+'/');
addpath('./export/');

%% data import & preprocessing
% Import RGC mosaics data from retinal wave dataset
load('./wavedata/'+WaveDataset+'/mosaics.mat');
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
wavecnt = load('./wavedata/'+WaveDataset+'/wavecnt.mat'); wavecnt = wavecnt.wavecnt-1;
waveset = cell(2,wavecnt); sig_wave = d_OFF*0.85; tic % Wave diffuse parameter, 1.3 for cat
wavedir = zeros(wavecnt,1);
crop_win = min(crop_x,crop_y);
ON_center = find(abs(pos_ON(:,1))<crop_win & abs(pos_ON(:,2))<crop_win);
OFF_center = find(abs(pos_OFF(:,1))<crop_win & abs(pos_OFF(:,2))<crop_win);
for ii = 1:wavecnt
    load('./wavedata/'+WaveDataset+'/wave'+num2str(ii)+'.mat');
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
imgsize_x = 200; % Filtered map width
img_sig = 9; % Gaussian image filtering width (unit: pixels)
img_sig = 7; % Gaussian image filtering width (unit: pixels)
ff_w0_sig = 18; % 24 for monkey/mouse, 18 for cat % Initial Exponential wiring range
ff_w0_str = 0.05; % Initial wiring strength
ff_w0_thr = 0; % V1 selection threshold
% V1 response curve
V1_thr = 0.5;
V1_slope = 0.15;
V1_max = 1;
V1_tau = 15; % Response averaging window

% Feedforward wiring update
ff_eps = 2e-3; % 1e-3 % 2e-9; % Learning rate
ff_w_sum_lim = 0.7; ff_w_lim = ff_w_sum_lim/5; % Resource limit
ff_w_normalize = 0.7; % Final normalization

% Horizontal connection learning
rnn_r_lim = 360;
rnn_eps = 2e-8; % Learning rate
rnn_w_sum_lim = 0.01; rnn_w_lim = rnn_w_sum_lim/20; % Resource limit

% Correlation pattern computation
% ROI
roi_x = [-620 220]; roi_y = [-420 420]; win_size = 420;
corr_size = 100;
% Random activity
rnn_w_normalize = 3; % Weight normalization
n_events = 200; % Number of random events
t_steps_max = 10; % Maximum integration steps
div_thr = 0.9; % Response divergence detection threshold
point_amp = 10; point_sig = 20; % Random point stimulus
noise_amp = 0.01; noise_sig = 30; % Background noise stimulus

%% model initialization
% Initial V1 sampling locations
if mouse == true
    dipole = 1; % if mouse, sample hexagonally based on retina/V1 sampling ratio
else
    dipole = 2; % if higher mammal, sample dipoles
end
pos_V1_init = init_V1_mosaic(d_V1,d_OFF,crop_x,crop_y,pos_ON,pos_OFF,dipole);

% Initial feedforward wiring
Result = init_feedforward(ff_w0_sig,pos_OFF,pos_ON,pos_V1_init,ff_w0_str,ff_w0_thr);
w0_V1_ON = cell2mat(Result(1));
w0_V1_OFF = cell2mat(Result(2));
pos_V1 = cell2mat(Result(3));

Result = init_horizontal(pos_V1,rnn_r_lim);
w0_V1_V1 = cell2mat(Result(1))*rnn_w_sum_lim;

% Save mosaic & initialized V1 network data
save(char(folderDir+"/mosaics.mat"),'pos_ON','pos_OFF','pos_V1_init','pos_V1');
save(char(folderDir+"/w0.mat"),'w0_V1_ON','w0_V1_OFF','w0_V1_V1');

% Illustrations of mosaics
figure; hold on; axis xy image; title("unit: um");
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
scatter(pos_OFF(:,1),pos_OFF(:,2),'b','filled');
scatter(pos_ON(:,1),pos_ON(:,2),'r','filled');
scatter(pos_V1(:,1),pos_V1(:,2),'og');

density_V1 = size(pos_V1,1)/4/crop_x/crop_y;
d_V1 = sqrt(2/sqrt(3)/density_V1);

%% RF formation by retinal wave
w_V1_ON = w0_V1_ON;
w_V1_OFF = w0_V1_OFF;
% To check 5 V1 cells in reduced time, set debug to true
debug = false;
% Test mode
V1_check = randi(size(w_V1_ON,1),5, 1);
if debug
    w_V1_ON = w_V1_ON(V1_check,:);
    w_V1_OFF = w_V1_OFF(V1_check,:);
end

% Set handles
wire = figure(100); wire.InvertHardcopy = 'off';
set(wire, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
myVideo = VideoWriter(char(folderDir+"/figure_learning/RGC_V1.avi"));
myVideo.FrameRate = 10;
open(myVideo); % Open movie frame

% RGC-V1 wiring refinement
for epoch = 1:50
Result = ff_Hebbian_update(epoch,wavecnt,waveset,w_V1_ON,w_V1_OFF,V1_max,V1_thr,V1_slope,V1_tau,ff_eps,ff_w_sum_lim,ff_w_lim);
w_V1_ON = cell2mat(Result(1));
w_V1_OFF = cell2mat(Result(2));
% Visualize current rewiring
for nn = 1:5
    V1_ind = V1_check(nn);
    subplot(2,5,nn); plot(1,1); hold on; title(num2str(V1_ind)+", initial");
    for jj = 1:size(pos_ON,1)
        if w0_V1_ON(V1_ind,jj) > 0
            scatter(pos_ON(jj,1),pos_ON(jj,2),30,w0_V1_ON(V1_ind,jj),'filled','markeredgecolor','r');
        end
    end
    for event = 1:size(pos_OFF,1)
        if w0_V1_OFF(V1_ind,event) > 0
            scatter(pos_OFF(event,1),pos_OFF(event,2),30,w0_V1_OFF(V1_ind,event),'filled','markeredgecolor','b');
        end
    end
    plot(pos_V1(V1_ind,1),pos_V1(V1_ind,2),'go');
    axis xy image; set(gca,'Color','k'); hold off; colormap(gray); caxis([0 max(w0_V1_ON(V1_ind,:))]);
    xlim([pos_V1(V1_ind,1)-200 pos_V1(V1_ind,1)+200]);
    ylim([pos_V1(V1_ind,2)-200 pos_V1(V1_ind,2)+200]);
    subplot(2,5,nn+5); plot(1,1); hold on; title(num2str(V1_ind)+", epoch "+num2str(epoch));
    if ~debug; nn = V1_ind; end
    for jj = 1:size(pos_ON,1)
        if w_V1_ON(nn,jj) > 0
            scatter(pos_ON(jj,1),pos_ON(jj,2),30,w_V1_ON(nn,jj),'filled','markeredgecolor','r');
        end
    end
    for event = 1:size(pos_OFF,1)
        if w_V1_OFF(nn,event) > 0
            scatter(pos_OFF(event,1),pos_OFF(event,2),30,w_V1_OFF(nn,event),'filled','markeredgecolor','b');
        end
    end
    plot(pos_V1(V1_ind,1),pos_V1(V1_ind,2),'go');
    axis xy image; set(gca,'Color','k'); hold off; colormap(gray); caxis([0 max(w_V1_ON(nn,:))]);
    xlim([pos_V1(V1_ind,1)-200 pos_V1(V1_ind,1)+200]);
    ylim([pos_V1(V1_ind,2)-200 pos_V1(V1_ind,2)+200]);
end
drawnow; writeVideo(myVideo,getframe(wire));
end
close(myVideo); % Close movie frame

% Final normalization
input_sum = sum(w_V1_ON,2)+sum(w_V1_OFF,2);
w_V1_ON = w_V1_ON./input_sum*ff_w_normalize;
w_V1_OFF = w_V1_OFF./input_sum*ff_w_normalize;
% Save RGC-V1 wiring data
save(char(folderDir+"/data_weight_matrix/w_V1_RGC"+".mat"),'pos_V1','w0_V1_ON','w0_V1_OFF','w_V1_ON','w_V1_OFF');

%% orientation preference
show_V1_inds = [100,200,300];
[~,show_V1_inds(1)] = min(abs(pos_V1(:,1)-266.7682));
[~,show_V1_inds(2)] = min(abs(pos_V1(:,1)+177.9865));
[~,show_V1_inds(3)] = min(abs(pos_V1(:,1)-44.3908));

wavenum = 38;
wave_ON = cell2mat(waveset(1,wavenum));
wave_OFF = cell2mat(waveset(2,wavenum));
figure;
subplot(121); hold on;
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
scatter(pos_ON(:,1),pos_ON(:,2),30,wave_ON(:,10),'filled','MarkerEdgeColor','r');
scatter(pos_OFF(:,1),pos_OFF(:,2),30,wave_OFF(:,10),'filled','MarkerEdgeColor','b');
colormap(gray); axis image; colorbar;
subplot(122); hold on;
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
scatter(pos_ON(:,1),pos_ON(:,2),30,wave_ON(:,20),'filled','MarkerEdgeColor','r');
scatter(pos_OFF(:,1),pos_OFF(:,2),30,wave_OFF(:,20),'filled','MarkerEdgeColor','b');
colormap(gray); axis image; colorbar;

% wave propagation
wave_ON = cell2mat(waveset(1,wavenum));
wave_OFF = cell2mat(waveset(2,wavenum));
V1_ff_input = w_V1_ON*wave_ON + w_V1_OFF*wave_OFF; % V1 X TIME
% Compute rate(t)
V1_rate = zeros(size(V1_ff_input,1), size(V1_ff_input,2)+1);
for tt = 1:size(V1_ff_input,2)
V1_rec_input = w0_V1_V1*V1_rate(:,tt); % V1 X TIME
V1_input = V1_ff_input(:,tt) + V1_rec_input;
V1_rate(:,tt+1) = V1_max*logsig((V1_input-V1_thr)/V1_slope); % V1 X TIME
end
figure; hold on;
plot(V1_rate(show_V1_inds(1),:),'r');
plot(V1_rate(show_V1_inds(2),:),'g');
plot(V1_rate(show_V1_inds(3),:),'b');

Result = compute_OP(pos_ON,pos_OFF,w0_V1_ON,w0_V1_OFF,w_V1_ON,w_V1_OFF);
op0 = Result(:,1);
op = Result(:,2);

% Dipole orientation across cortical surface
figure; suptitle("OP");
subplot(121); hold on; axis xy image;
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
scatter(pos_V1(:,1),pos_V1(:,2),30,op0,'filled'); colormap(hsv);
caxis([-pi/2 pi/2]); colorbar;
subplot(122); hold on; axis xy image;
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
plot(pos_V1(show_V1_inds(1),1),pos_V1(show_V1_inds(1),2),'ro',"linewidth",3,"MarkerSize", 20);
plot(pos_V1(show_V1_inds(2),1),pos_V1(show_V1_inds(2),2),'go',"linewidth",3,"MarkerSize", 20);
plot(pos_V1(show_V1_inds(3),1),pos_V1(show_V1_inds(3),2),'bo',"linewidth",3,"MarkerSize", 20);
scatter(pos_V1(:,1),pos_V1(:,2),30,op,'filled'); colormap(hsv);
caxis([-pi/2 pi/2]); colorbar;

imgsize_y = imgsize_x*crop_y/crop_x;
opmap0 = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_V1,op0,true);
opmap = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_V1,op,true);

% OP map before/after RGC-V1 refinement
figure;
subplot(121); imagesc(opmap0); axis xy image; title('Initial'); colormap(hsv);
caxis([-pi/2 pi/2]); colorbar;
subplot(122); imagesc(opmap); axis xy image; title('Final'); colormap(hsv);
caxis([-pi/2 pi/2]); colorbar; drawnow;

% OP histogram before/after RGC-V1 refinement
figure;
GMModel1 = fitgmdist(op0*180/pi,3);
GMModel2 = fitgmdist(op*180/pi,3);
subplot(121); hold on;
xticks([-90 -60 -30 0 30 60 90]); xlim([-90 90]);
histogram(op0*180/pi,-90:10:90,'Normalization','pdf');
fplot(@(x)pdf(GMModel1,x),[-90 90],'r');
histogram(op*180/pi,-90:10:90,'Normalization','pdf');
fplot(@(x)pdf(GMModel2,x),[-90 90],'b');
del_op = abs(op-op0);
del_op(del_op>pi/2) = pi-del_op(del_op>pi/2);
del_op(del_op<-pi/2) = -pi-del_op(del_op<-pi/2);
subplot(122); histogram(del_op*180/pi,0:10:90,'Normalization','pdf'); drawnow;

%% Map period
[y_size, x_size] = size(opmap);
map_size = max(y_size,x_size);
map_cen = (map_size+1)/2;
comp_map = exp(2i.*opmap);
fft_map = abs(fftshift(fft2(comp_map-mean(comp_map(:)))));


xcen = (x_size+1)/2;
ycen = (y_size+1)/2;
dist2 = ((1:y_size)'-ycen).^2+((1:x_size)-xcen).^2;
y_arr = repmat((1:y_size)', [1, x_size]);
x_arr = repmat((1:x_size)'', [y_size, 1]);

max_mask = imregionalmax(fft_map);
max_mask_flat = max_mask(:);
dist2 = dist2.*max_mask; dist2 = dist2(:); dist2(max_mask_flat==0) = [];
y_arr = y_arr.*max_mask; y_arr = y_arr(:); y_arr(max_mask_flat==0) = [];
x_arr = x_arr.*max_mask; x_arr = x_arr(:); x_arr(max_mask_flat==0) = [];

[~, idx] = sort(dist2);
peak_idx = idx(1);

y_max = y_arr(peak_idx);
x_max = x_arr(peak_idx);
peak_dist = abs((y_max-ycen) + 1i*(x_max-xcen));
period_pixel = map_size/peak_dist;

period_retina = period_pixel / y_size * crop_y*2;
period_V1 = period_retina * retina_V1_ratio;

%% Hebbian learning by retinal wave
w_V1_V1 = w0_V1_V1;
wp_V1_V1 = w0_V1_V1; % Permuted retinal wave
% 
% save(char(folderDir+"/data_weight_matrix/w_epoch0.mat"),'w_V1_V1','wp_V1_V1','epoch');
for epoch = 1:30
%      w_V1_V1 = V1_Hebbian_update(epoch,wavecnt,waveset,w_V1_ON,w_V1_OFF,w_V1_V1,...
%          V1_max,V1_thr,V1_slope,V1_tau,rnn_eps,rnn_w_sum_lim,rnn_w_lim,false);
    wp_V1_V1 = V1_Hebbian_update(epoch,wavecnt,waveset,w_V1_ON,w_V1_OFF,wp_V1_V1,...
        V1_max,V1_thr,V1_slope,V1_tau,rnn_eps,rnn_w_sum_lim,rnn_w_lim,true);
    gpuDevice(1); % Refresh GPU memory
    save(char(folderDir+"/data_weight_matrix/w_epoch"+num2str(epoch)+".mat"),'w_V1_V1','wp_V1_V1','epoch');
    drawnow;
end

%% horizontal network analysis
w_analysis = w_V1_V1; % connectivity analysis
[~,ref_V1] = min(abs(pos_V1(:,1)-200.1708));
ref_pos = round(pos_V1(ref_V1,:)*imgsize_x/crop_x/2+[imgsize_x,imgsize_y]/2);
% RGC-V1 plot
figure; imagesc(opmap); axis xy image;
title('final'); colormap(hsv); caxis([-pi/2 pi/2]); colorbar; hold on;
plot(ref_pos(1),ref_pos(2),"bo","linewidth",3,"Markersize",5); drawnow;
% Clustered synapses on orientation domains
learn_map = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_V1,w_analysis(:,ref_V1),false);
diff = opmap-opmap(ref_pos(2),ref_pos(1));
diff = abs(diff); diff(diff>pi/2) = pi-diff(diff>pi/2);
img_diff = ind2rgb(round(diff/max(max(diff))*255),gray(255));
n_syn = round(size(w0_V1_V1,1)*0.1);
[~,ind_V1] = sort(w_analysis(:,ref_V1),'descend');
connect_V1 = (pos_V1(ind_V1(1:n_syn),:)+[crop_x crop_y])/2/crop_x*imgsize_x;

% Plot import and merge
figure; subplot(121); hold on;
image(img_diff); axis image xy; colormap(gray);
plot(ref_pos(1),ref_pos(2),"bo","linewidth",3,"Markersize",5);
plot(connect_V1(:,1),connect_V1(:,2),'go', 'linewidth', 2);
figure; [xx,yy] = meshgrid(1:size(opmap,2),1:size(opmap,1));
[C_op,~] = contour(xx,yy,diff-pi/8);
[xC_op,yC_op,zC_op] = C2xyz(C_op); xC_op(zC_op~=0)=[]; yC_op(zC_op~=0)=[]; zC_op(zC_op~=0)=[]; close;
for ii = 1:length(zC_op)
    plot(cell2mat(xC_op(ii)),cell2mat(yC_op(ii)),'-k','linewidth',1.5);
end

subplot(122); hold on;
imagesc(opmap); axis image xy; colormap(hsv);
scatter(connect_V1(:,1),connect_V1(:,2),50, op(ind_V1(1:n_syn)),'filled'); drawnow;

% OP-weight trend test
figure;
subplot(131); analysis_OP_weight_hist(op,w0_V1_V1,pos_V1);
subplot(132); analysis_OP_weight_hist(op,w_V1_V1,pos_V1);
subplot(133); analysis_OP_weight_hist(op,wp_V1_V1,pos_V1);

% Spatial clustering analysis: Hopkins' ratio statistics
CI_initial = analysis_clustering_Hopkins(pos_V1,w0_V1_V1,d_OFF,crop_x,crop_y);
CI_clustered = analysis_clustering_Hopkins(pos_V1,w_V1_V1,d_OFF,crop_x,crop_y);
CI_permuted = analysis_clustering_Hopkins(pos_V1,wp_V1_V1,d_OFF,crop_x,crop_y);

figure;
subplot(121);
[h,p] = ttest2(CI_initial',CI_clustered');
hold on; bar([1 2],[mean(CI_initial),mean(CI_clustered)]);
errorbar([1 2],[mean(CI_initial),mean(CI_clustered)],[std(CI_initial),std(CI_clustered)],'.k');
title(sprintf("Clustering index (CI), N = "+num2str(size(CI_initial,2))+", t-test p = "+num2str(p))); drawnow;
ylim([-0.5 1.5]);
subplot(122);
[h,p] = ttest2(CI_initial',CI_permuted');
hold on; bar([1 2],[mean(CI_initial),mean(CI_permuted)]);
errorbar([1 2],[mean(CI_initial),mean(CI_permuted)],[std(CI_initial),std(CI_permuted)],'.k');
title(sprintf("Permuted wave, N = "+num2str(size(CI_initial,2))+ ", t-test p = "+num2str(p))); drawnow;
ylim([-0.5 1.5]);

%% activity correlation computation
ind_V1_corr = find(pos_V1(:,1)>roi_x(1)&pos_V1(:,1)<roi_x(2)&pos_V1(:,2)>roi_y(1)&pos_V1(:,2)<roi_y(2));
pos_V1_corr = pos_V1(ind_V1_corr,:);
pos_V1_corr = pos_V1_corr - (min(pos_V1_corr) + max(pos_V1_corr))/2;
w_V1_V1_corr = w_V1_V1(ind_V1_corr,ind_V1_corr);
corr_sig = img_sig*crop_x/win_size*corr_size/imgsize_x;
opmap_corr = V1_filt_Gaussian(win_size,win_size,corr_size,corr_sig,pos_V1_corr,op(ind_V1_corr),true);
figure; imagesc(opmap_corr); colormap(hsv); colorbar; axis image xy;

w_V1_V1_corr = rnn_w_normalize*w_V1_V1_corr./sum(w_V1_V1_corr,1);

% Generate random stimulus
V1_point_input = point_amp*point_stimulus(pos_V1_corr,point_sig,n_events);
V1_noise_input = noise_amp*background_noise_stimulus(pos_V1_corr,noise_sig,n_events);

% Obtain and filter V1 responses
V1_profile = V1_stochastic_response(n_events,t_steps_max,div_thr,V1_max,V1_slope,V1_thr,...
    w_V1_V1_corr,V1_point_input,V1_noise_input);
V1_profile_filtered = zeros(corr_size*corr_size,n_events);
for event = 1:n_events; fprintf("%d\n",event);
    show_img = mod(event,20)==0;
    temp = V1_filt_Gaussian(win_size,win_size,corr_size,corr_sig,pos_V1_corr,V1_profile(:,event),true);
    V1_profile_filtered(:,event) = reshape(temp,[corr_size*corr_size 1]);
end

% Crop and normalize images
V1_profile_crop = reshape(V1_profile_filtered,corr_size,corr_size,n_events);
V1_profile_crop = reshape(V1_profile_crop,[corr_size*corr_size n_events]);
V1_profile_norm = (V1_profile_crop-mean(V1_profile_crop,1))./std(V1_profile_crop,1);
V1_profile_norm(isnan(V1_profile_norm)) = 0;
% Compute activity correlation pattern
corr_mat = gather(reshape(corr(V1_profile_norm','Type','Pearson'),[corr_size corr_size corr_size corr_size]));
save(char(folderDir+"/corr_epoch30.mat"),'corr_mat');

% Moving reference point
corr_fig = figure(77); xx = round(corr_size/2);
myVideo = VideoWriter(char(folderDir+"/figure_spontaneous_activity/ActivityCorrPattern_step_"+num2str(epoch) +".avi"));
myVideo.FrameRate = 10; open(myVideo);
for yy = 1:corr_size
    a = subplot(121); imagesc(opmap_corr); colormap(a,hsv); colorbar; caxis([-pi/2 pi/2]);
    hold on; plot(xx,yy,'ko','Linewidth',2); axis xy image; drawnow;
    b = subplot(122); imagesc(squeeze(corr_mat(yy,xx,:,:))); colormap(b,redblue); colorbar; caxis([-0.75 0.75]);
    hold on; plot(xx,yy,'ko','Linewidth',2); axis image xy; set(gcf,'units','points','position',[300,100,1000,400]); drawnow;
    writeVideo(myVideo,getframe(corr_fig));
end; close; close(myVideo); % Close movie frame

%% activity correlation matching
% Compute correlation coefficient between orientation and correlation patterns
R_all = [];
for ref_V1 = 1:size(pos_V1_corr,1)
    try
    ref_pos = round((pos_V1_corr(ref_V1,:)*corr_size/win_size/2+[corr_size,corr_size])/2);
    corr_map = squeeze(corr_mat(ref_pos(2),ref_pos(1),:,:)); % Correlation pattern
    % Orientation-activity correlation
    opdiff = abs(opmap_corr-opmap_corr(ref_pos(2),ref_pos(1)));
    opdiff(opdiff>pi/2) = pi-opdiff(opdiff>pi/2);
    opsim = (pi/2-opdiff)/max(opdiff(:)); % Scaled OP similarity map
    opsim_1d = opsim(:);
    corr_map_1d = corr_map(:);
    [R,P] = corrcoef(opsim_1d,corr_map_1d);
    R_all = [R_all R(1,2)];
    catch
        disp("invalid")
    end
end
[h,p,ci,stats] = ttest(R_all)

figure;
subplot(121); scatter(ones(1,size(R_all,2)),R_all);
hold on; errorbar(2,mean(R_all),std(R_all)); xlim([0.5 2.5]); ylim([-1 1]);
subplot(122); histogram(R_all);
suptitle("act-ori corr, for all V1 reference points: "...
    + num2str(mean(R_all)) + "+-" + num2str(std(R_all)));

[R_sort, R_ind] = sort(R_all,'descend');
ref_V1 = R_ind(24);
[~,ref_V1] = min(abs(pos_V1(:,1)+177));

ref_pos = round((pos_V1(ref_V1,:)*corr_size/win_size/2+[corr_size,corr_size])/2);
corr_map = squeeze(corr_mat(ref_pos(2),ref_pos(1),:,:)); % Correlation pattern
% Orientation-activity correlation
opdiff = abs(opmap_corr-opmap_corr(ref_pos(2),ref_pos(1)));
opdiff(opdiff>pi/2) = pi-opdiff(opdiff>pi/2);
opsim = (pi/2-opdiff)/max(opdiff(:)); % Scaled OP similarity map
opsim_1d = opsim(:); corr_map_1d = corr_map(:); [R,P] = corrcoef(opsim_1d,corr_map_1d);

% Statistical significance of orientation-activity correlation: random rotated control
[mean1,mean2,std1,std2,p] = analysis_CORR_OPSIM_significance(corr_map,opsim,100);
% Plot with contours
opdiff_map = abs(opmap_corr-opmap_corr(ref_pos(2),ref_pos(1)));
opdiff_map(opdiff_map>pi/2) = pi-opdiff_map(opdiff_map>pi/2);
figure; [xx,yy] = meshgrid(1:size(opmap_corr,2),1:size(opmap_corr,1));
[C_op,~] = contour(xx,yy,opdiff_map-pi/4);
[xC_op,yC_op,zC_op] = C2xyz(C_op); xC_op(zC_op~=0)=[]; yC_op(zC_op~=0)=[]; zC_op(zC_op~=0)=[]; close;
figure; [xx,yy] = meshgrid(1:size(corr_map,2),1:size(corr_map,1));
[C_corr,~] = contour(xx,yy,corr_map);
[xC_c,yC_c,zC_c] = C2xyz(C_corr); xC_c(zC_c~=0)=[]; yC_c(zC_c~=0)=[]; zC_c(zC_c~=0)=[]; close;
figure;
a = subplot(151); imagesc(corr_map); colormap(a,redblue); caxis([-0.75 0.75]); colorbar('southoutside');
axis image xy; hold on; plot(ref_pos(1),ref_pos(2),'ko','Linewidth',2);
for ii = 1:length(zC_c)
    plot(cell2mat(xC_c(ii)),cell2mat(yC_c(ii)),'-g','linewidth',1.5);
end
b = subplot(152); imagesc(opmap_corr); colormap(b,hsv); colorbar('southoutside');
axis image xy; hold on; plot(ref_pos(1),ref_pos(2),'ko','Linewidth',2);
for ii = 1:length(zC_op)
    plot(cell2mat(xC_op(ii)),cell2mat(yC_op(ii)),'-k','linewidth',1.5);
end
c = subplot(153); imagesc(opsim); colormap(c,gray); caxis([0 1]); colorbar('southoutside');
axis image xy; hold on; plot(ref_pos(1),ref_pos(2),'ko','Linewidth',2);
for ii = 1:length(zC_op)
    plot(cell2mat(xC_op(ii)),cell2mat(yC_op(ii)),'-k','linewidth',1.5);
end
for ii = 1:length(zC_c)
    plot(cell2mat(xC_c(ii)),cell2mat(yC_c(ii)),'-g','linewidth',1.5);
end
subplot(154);
hold on; errorbar([1,2],[mean1,mean2],[std1,std2]); ylim([-0.2 1]); xlim([0.5 2.5]);
title(sprintf("Model pattern corr, t-test p = "+num2str(p))); drawnow;
set(gcf,'units','points','position',[300,100,1000,400]);
suptitle(sprintf("Figure 3: Activity-orientation matching"+...
    "\nN = "+num2str(size(opsim,1))+", r = "+num2str(R(1,2))+", p = "+num2str(P(1,2)))); drawnow;

%% For many reference points, with control
R_mean_all = [];
control_mean_all = [];
for ref_V1 = 1:size(pos_V1_corr,1)
    try
    ref_pos = round((pos_V1_corr(ref_V1,:)*corr_size/win_size/2+[corr_size,corr_size])/2);
    corr_map = squeeze(corr_mat(ref_pos(2),ref_pos(1),:,:)); % Correlation pattern
    % Orientation-activity correlation
    opdiff = abs(opmap_corr-opmap_corr(ref_pos(2),ref_pos(1)));
    opdiff(opdiff>pi/2) = pi-opdiff(opdiff>pi/2);
    opsim = (pi/2-opdiff)/max(opdiff(:)); % Scaled OP similarity map
    
    [mean1,mean2,std1,std2,p] = analysis_CORR_OPSIM_significance(corr_map,opsim,100);
    disp([mean1 mean2]);
    R_mean_all(ref_V1) = mean1;
    control_mean_all(ref_V1) = mean2;
    catch
        disp("invalid")
    end
end
% Paired t-test
[h,p] = ttest(R_mean_all,control_mean_all);

mean1 = mean(R_mean_all);
mean2 = mean(control_mean_all);
std1 = std(R_mean_all);
std2 = std(control_mean_all);
disp(mean1);
disp(mean2);

figure;
subplot(121); scatter(ones(1,size(R_mean_all,2)),R_mean_all);
hold on; errorbar(1,mean(R_mean_all),std(R_mean_all)); xlim([0.5 2.5]); ylim([-1 1]);
scatter(ones(1,size(control_mean_all,2))*2,control_mean_all);
errorbar(2,mean(control_mean_all),std(control_mean_all)); xlim([0.5 2.5]); ylim([-1 1]);
title("paired t-test p= " + num2str(p))
subplot(122); histogram(R_mean_all);
suptitle("act-ori corr, for all V1 reference points: "...
    + num2str(mean(R_mean_all)) + "+-" + num2str(std(R_mean_all)));

%% input-output images
myVideo = VideoWriter(char(folderDir+"/figure_spontaneous_activity/corr.avi"));
myVideo.FrameRate = 5; open(myVideo);
profile = figure(77);
for event = 1:n_events
    img_input = V1_filt_Gaussian(win_size,win_size,corr_size,corr_sig,pos_V1_corr,V1_point_input(:,event)+V1_noise_input(:,event),false);
    img_profile = reshape(V1_profile_norm(:,event),[corr_size corr_size]);
    clf;
    subplot(121); imagesc(img_input); axis image xy; colorbar('southoutside');
    subplot(122); imagesc(img_profile); axis image xy; colorbar('southoutside');
    suptitle(['Normalized activity ' num2str(event)]); colormap gray; drawnow;
    saveas(profile,char(folderDir+"/figure_spontaneous_activity/num"+num2str(event)+".fig"));
    writeVideo(myVideo,getframe(profile));
end; close; close(myVideo); % Close movie frame
