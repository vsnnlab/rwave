%% Execution environment
clear all; close all; % close(myVideo);
parallel.gpu.rng(0, 'Philox4x32-10');
env_run = true; env_save_wave = true; env_save_movie = true;
% Paths
addpath('./init_model/'); addpath('./function_wave/'); addpath('./export/');
env_dir = "./export/test";
if env_run; env_dir = "./export/"+datestr(now,'yyyy-mm-dd,HH-MM'); end
if env_save_wave || env_save_movie; mkdir(env_dir); end

%% Import data mosaic
data_mosaic = load("init_model/stat_M623_scaled.mat");
data_OFF = data_mosaic.mosaicOFF{1};
data_ON = data_mosaic.mosaicON{1};

data_OFF = [real(data_OFF) imag(data_OFF)];
data_ON = [real(data_ON) imag(data_ON)];

% Zero-center
data_center = (min([data_OFF;data_ON])+max([data_OFF;data_ON]))/2;
data_OFF = data_OFF-data_center;
data_ON = data_ON-data_center;

% Crop
crop_x = 400;
crop_y = 400;
crop_ON = data_ON; crop_OFF = data_OFF;
crop_ON(abs(crop_ON(:,1))>crop_x|abs(crop_ON(:,2))>crop_y,:) = [];
crop_OFF(abs(crop_OFF(:,1))>crop_x|abs(crop_OFF(:,2))>crop_y,:) = [];
win_export = 1000;

%% Model parameters
n_waves = 2000;
dt = 0.1;
period = 10; steps = period/dt;
pad_r = 1900;

% Asynchronous / synchronous wave option
synchronize = false;

% Cell model
C = 1; delC = 0.2; % Randomly varying cell coupling
period_fire = 1; steps_fire = period_fire/dt;
f = 0.8; % Recruitable cell fraction
start_margin = 400; start_size = 400; % Wave initiation

if ~synchronize
    r_RGC = 400; % ON RGC dendritic radius
    r_AC = 40; % AC dendritic radius
    thr_ON = 7; thr_AC = 7; thr_OFF = -0.1;
else
    r_RGC = 400; % ON RGC dendritic radius
    r_AC = 40; % AC dendritic radius
    thr_ON = 7; thr_OFF = 7;
end
    
%% Retinal network model
% Synthesize padding mosaic
Result = RGC_AC_mosaic(crop_ON,crop_OFF,pad_r,crop_x,crop_y);
syn_ON = cell2mat(Result(1));
syn_OFF = cell2mat(Result(2));
pos_AC = cell2mat(Result(3));
d_ON = cell2mat(Result(4));
d_OFF = cell2mat(Result(5));

% Mark number of data RGC
mark_ON = size(crop_ON,1);
mark_OFF = size(crop_OFF,1);

% Concat data and padding RGC
pos_ON = [crop_ON;syn_ON];
pos_OFF = [crop_OFF;syn_OFF];

% Mark RGCs to record retinal wave (cropped data + some synthesized margin)
inds_ON_save = find(abs(pos_ON(:,1))<win_export&abs(pos_ON(:,2))<win_export);
inds_OFF_save = find(abs(pos_OFF(:,1))<win_export&abs(pos_OFF(:,2))<win_export);
saved_ON = pos_ON(inds_ON_save,:);
saved_OFF = pos_OFF(inds_OFF_save,:);

% Weight matrix (binary)
if ~synchronize
    % async: ON -> AC -| OFF
    w_ON_ON = init_conn(pos_ON,pos_ON,r_RGC); % ON <- ON
    w_ON_ON = w_ON_ON-diag(diag(w_ON_ON)); % No autosynaptic connection
    w_AC_ON = init_conn(pos_ON,pos_AC,r_RGC); % AC <- ON
    w_OFF_AC = -init_conn(pos_AC,pos_OFF,r_AC); % OFF |- AC
else
    % sync: ON -> OFF
    w_ON_ON = init_conn(pos_ON,pos_ON,r_RGC); % ON <- ON
    w_OFF_ON = init_conn(pos_ON,pos_OFF,r_RGC); % OFF <- ON
end

figure(1);
subplot(131); hold on; axis xy image; title("Zhan & Troy 2000, unit: um");
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
scatter(data_OFF(:,1),data_OFF(:,2),'b','filled');
scatter(data_ON(:,1),data_ON(:,2),'r','filled');

subplot(132); hold on; axis xy image; title("Synthesized mosaics, unit: um");
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
scatter(pos_AC(:,1),pos_AC(:,2),'MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','none');
scatter(pos_OFF(:,1),pos_OFF(:,2),'c','filled');
scatter(pos_ON(:,1),pos_ON(:,2),'m','filled');
scatter(crop_OFF(:,1),crop_OFF(:,2),'b','filled');
scatter(crop_ON(:,1),crop_ON(:,2),'r','filled');

subplot(133); hold on; axis xy image; title("Saved mosaics, unit: um");
rectangle('Position',[-crop_x,-crop_y,2*crop_x,2*crop_y],'faceColor','k');
scatter(saved_OFF(:,1),saved_OFF(:,2),'c','filled');
scatter(saved_ON(:,1),saved_ON(:,2),'m','filled');
scatter(saved_OFF(1:mark_OFF,1),saved_OFF(1:mark_OFF,2),'b','filled');
scatter(saved_ON(1:mark_ON,1),saved_ON(1:mark_ON,2),'r','filled');

% Save parameters
save(char(env_dir+"/mosaics.mat"),'d_ON','d_OFF','crop_ON','crop_OFF','crop_x','crop_y',...
    'saved_ON','saved_OFF','mark_ON','mark_OFF');

%% Retinal wave model
wavecnt = 0; sig_wave = d_OFF*0.85;
for ii = 1:n_waves
% 1 wave propagation under restricted dynamics
tic; fprintf("Wave %d\n",ii); save_ii = true;
Result = wave_initialize(pos_ON,pos_OFF,pos_AC,f,steps,steps_fire,pad_r-start_margin,start_size);
state_ON = cell2mat(Result(1)); state_AC = cell2mat(Result(2)); state_OFF = cell2mat(Result(3));
wave_ON = wave_filter(pos_ON,state_ON,sig_wave);
wave_OFF = wave_filter(pos_OFF,state_OFF,sig_wave);
wave_AC = wave_filter(pos_AC,state_AC*2,sig_wave);
if env_save_movie && ii<=10
    myVideo = VideoWriter(char(env_dir+"/trial"+num2str(ii)+".avi"));
    myVideo.FrameRate = 1/dt; open(myVideo);
    handle = figure(999);
    set(gcf,'Position',[100, 100, 1600, 500]);
    colormap(gray); suptitle("0 ms");
    
    subplot(131); hold on; axis xy image; title("ON"); colorbar; caxis([0 1]);
    scatter(pos_ON(:,1),pos_ON(:,2),10,wave_ON(:,1),'filled');
    rectangle('Position',[-crop_x -crop_y 2*crop_x 2*crop_y],"edgecolor","y","linewidth",1);
    
    subplot(132); hold on; axis xy image; title("OFF"); colorbar; caxis([0 1]);
    scatter(pos_OFF(:,1),pos_OFF(:,2),10,wave_OFF(:,1),'filled');
    rectangle('Position',[-crop_x -crop_y 2*crop_x 2*crop_y],"edgecolor","y","linewidth",1);
    
    subplot(133); hold on; axis xy image; title("AC"); colorbar; caxis([-1 2]);
    scatter(pos_AC(:,1),pos_AC(:,2),10,state_AC(:,1),'filled');
    rectangle('Position',[-crop_x -crop_y 2*crop_x 2*crop_y],"edgecolor","y","linewidth",1);
    
    drawnow; writeVideo(myVideo,getframe(handle));
    saveas(handle,char(env_dir+"/step"+num2str(1)+".fig"));
end
for step = 2:steps
    if ~synchronize
        % Async wave
        Result = async_propagate(step,state_ON,state_AC,state_OFF,w_ON_ON,w_AC_ON,w_OFF_AC,thr_ON,thr_AC,thr_OFF,steps_fire);
        state_ON = cell2mat(Result(1)); state_AC = cell2mat(Result(2)); state_OFF = cell2mat(Result(3));
    else
        % Sync wave
        Result = sync_propagate(step,state_ON,state_OFF,w_ON_ON,w_OFF_ON,thr_ON,thr_OFF,steps_fire);
        state_ON = cell2mat(Result(1)); state_OFF = cell2mat(Result(2));
    end
    wave_ON = wave_filter(pos_ON,state_ON,sig_wave);
    wave_OFF = wave_filter(pos_OFF,state_OFF,sig_wave);
    wave_AC = wave_filter(pos_AC,state_AC*2,sig_wave);
    if env_save_movie && ii<=10
        handle = figure(999); clf;
        colormap(gray); suptitle(num2str(step*100-100)+" ms");
        
        subplot(131); hold on; axis xy image; title("ON"); colorbar; caxis([0 1]);
        scatter(pos_ON(:,1),pos_ON(:,2),10,wave_ON(:,step),'filled');
        rectangle('Position',[-crop_x -crop_y 2*crop_x 2*crop_y],"edgecolor","y","linewidth",1);
        
        subplot(132); hold on; axis xy image; title("OFF"); colorbar; caxis([0 1]);
        scatter(pos_OFF(:,1),pos_OFF(:,2),10,wave_OFF(:,step),'filled');
        rectangle('Position',[-crop_x -crop_y 2*crop_x 2*crop_y],"edgecolor","y","linewidth",1);
        
        subplot(133); hold on; axis xy image; title("AC"); colorbar; caxis([-1 2]);
        scatter(pos_AC(:,1),pos_AC(:,2),10,state_AC(:,step),'filled');
        rectangle('Position',[-crop_x -crop_y 2*crop_x 2*crop_y],"edgecolor","y","linewidth",1);
        
        drawnow; writeVideo(myVideo,getframe(handle));
        saveas(handle,char(env_dir+"/step"+num2str(step)+".fig"));
    end
    if sum(state_ON(:,step)==2)==0 % If no ON cell is firing, terminate
        if step<=20; save_ii = false; end % If wavefront is not formed, don't save
        break;
    end
end
% Export wave data
if save_ii && env_save_wave
    wavecnt = wavecnt+1;
    wave_save(env_dir,wavecnt,state_ON(inds_ON_save,:),state_OFF(inds_OFF_save,:));
    matfile = fullfile(env_dir,"wavecnt.mat"); save(matfile,'wavecnt');
end
gpuDevice(1); if env_save_movie && ii<=10; close(myVideo); end; toc
end
