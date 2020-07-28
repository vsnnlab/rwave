%% Model pattern 1
N = 12; DATA_CMAP = imread(['Data_autocorr/AUTOCORR_',num2str(N),'.jpg']);
C_MAP_MODEL1 = function_RB_C_mapping(DATA_CMAP,true)*0.75; close; close;
% Define region of interest
num_ROI = 4; ROI_SIZE = 60;
X0 = [120 25 120 25]; Y0 = [150 150 50 50];
MEAN_IMAGE1 = function_autocorrelation_analysis(num_ROI,X0,Y0,ROI_SIZE,C_MAP_MODEL1,5);
PEAK = FastPeakFind(MEAN_IMAGE1*100,0.1);
[xx,yy] = meshgrid(1:size(MEAN_IMAGE1,2),1:size(MEAN_IMAGE1,1));
figure; [C1,~] = contour(xx,yy,MEAN_IMAGE1-0.33); close;
[xC1,yC1,zC1] = C2xyz(C1); xC1(zC1~=0)=[]; yC1(zC1~=0)=[]; zC1(zC1~=0)=[];
figure; imagesc(MEAN_IMAGE1); axis image xy; colormap(jet); colorbar;
hold on; plot(PEAK(1:2:end),PEAK(2:2:end),'r+');
for ii = 1:length(zC1)
    plot(cell2mat(xC1(ii)),cell2mat(yC1(ii)),'-w','linewidth',1.5);
end
title("MH model 1");
% Define region of interest
num_ROI = 4; ROI_SIZE = 70; X0 = [570 570 475 475]; Y0 = [150 50 150 50];
MEAN_IMAGE2 = function_autocorrelation_analysis(num_ROI,X0,Y0,ROI_SIZE,C_MAP_MODEL1,5);
PEAK = FastPeakFind(MEAN_IMAGE2*100,0.1);
[xx,yy] = meshgrid(1:size(MEAN_IMAGE2,2),1:size(MEAN_IMAGE2,1));
figure; [C1,~] = contour(xx,yy,MEAN_IMAGE2-0.33); close;
[xC1,yC1,zC1] = C2xyz(C1); xC1(zC1~=0)=[]; yC1(zC1~=0)=[]; zC1(zC1~=0)=[];
figure; imagesc(MEAN_IMAGE2); axis image xy; colormap(jet); colorbar;
hold on; plot(PEAK(1:2:end),PEAK(2:2:end),'r+');
for ii = 1:length(zC1)
    plot(cell2mat(xC1(ii)),cell2mat(yC1(ii)),'-w','linewidth',1.5);
end
title("MH model 2");
% Define region of interest
num_ROI = 4; ROI_SIZE = 70; X0 = [1010 1010 914 914]; Y0 = [150 50 150 50];
MEAN_IMAGE3 = function_autocorrelation_analysis(num_ROI,X0,Y0,ROI_SIZE,C_MAP_MODEL1,5);
PEAK = FastPeakFind(MEAN_IMAGE3*100,0.1);
[xx,yy] = meshgrid(1:size(MEAN_IMAGE3,2),1:size(MEAN_IMAGE3,1));
figure; [C1,~] = contour(xx,yy,MEAN_IMAGE3-0.33); close;
[xC1,yC1,zC1] = C2xyz(C1); xC1(zC1~=0)=[]; yC1(zC1~=0)=[]; zC1(zC1~=0)=[];
figure; imagesc(MEAN_IMAGE3); axis image xy; colormap(jet); colorbar;
hold on; plot(PEAK(1:2:end),PEAK(2:2:end),'r+');
for ii = 1:length(zC1)
    plot(cell2mat(xC1(ii)),cell2mat(yC1(ii)),'-w','linewidth',1.5);
end
title("MH model 3");
%% Model pattern 2
N = 11; DATA_CMAP = imread(['Data_autocorr/AUTOCORR_',num2str(N),'.jpg']);
C_MAP_MODEL2 = function_RB_C_mapping(DATA_CMAP,true)*0.75; close; close;
% Define region of interest
num_ROI = 2; ROI_SIZE = 90; X0 = [83 83]; Y0 = [68 513];
MEAN_IMAGE = function_autocorrelation_analysis(num_ROI,X0,Y0,ROI_SIZE,C_MAP_MODEL2,5);
PEAK = FastPeakFind(MEAN_IMAGE*100,0.1);
[xx,yy] = meshgrid(1:size(MEAN_IMAGE,2),1:size(MEAN_IMAGE,1));
figure; [C1,~] = contour(xx,yy,MEAN_IMAGE-0.33); close;
[xC1,yC1,zC1] = C2xyz(C1); xC1(zC1~=0)=[]; yC1(zC1~=0)=[]; zC1(zC1~=0)=[];
figure; imagesc(MEAN_IMAGE); axis image xy; colormap(jet); colorbar;
hold on; plot(PEAK(1:2:end),PEAK(2:2:end),'r+');
for ii = 1:length(zC1)
    plot(cell2mat(xC1(ii)),cell2mat(yC1(ii)),'-w','linewidth',1.5);
end
title("MH model 4");
%% Data pattern 1
N = 2; DATA_CMAP = imread(['Data_autocorr/AUTOCORR_',num2str(FIG_NUM(N)),'.jpg']);
C_MAP_DATA1 = function_RB_C_mapping(DATA_CMAP,true)*0.75; close; close;
% Define region of interest
num_ROI = 4; ROI_SIZE = 210; X0 = [50 410 770 1150]; Y0 = [10 10 10 10];
MEAN_IMAGE = function_autocorrelation_analysis(num_ROI,X0,Y0,ROI_SIZE,C_MAP_DATA1,5);
PEAK = FastPeakFind(MEAN_IMAGE*100,0.1);
[xx,yy] = meshgrid(1:size(MEAN_IMAGE,2),1:size(MEAN_IMAGE,1));
figure; [C1,~] = contour(xx,yy,MEAN_IMAGE-0.33); close;
[xC1,yC1,zC1] = C2xyz(C1); xC1(zC1~=0)=[]; yC1(zC1~=0)=[]; zC1(zC1~=0)=[];
figure; imagesc(MEAN_IMAGE); axis image xy; colormap(jet); colorbar;
hold on; plot(PEAK(1:2:end),PEAK(2:2:end),'w+','linewidth',1.);
for ii = 1:length(zC1)
    plot(cell2mat(xC1(ii)),cell2mat(yC1(ii)),'-w','linewidth',1.5);
end
title("Individual 1");
%% Data pattern 2
N = 13; DATA_CMAP = imread(['Data_autocorr/AUTOCORR_',num2str(N),'.jpg']);
C_MAP_DATA2 = function_RB_C_mapping(DATA_CMAP,true)*0.75; close; close;
% Define region of interest
num_ROI = 4; ROI_SIZE = 140; X0 = [1 210 420 630]; Y0 = [70 70 70 70];
MEAN_IMAGE = function_autocorrelation_analysis(num_ROI,X0,Y0,ROI_SIZE,C_MAP_DATA2,1.1);
PEAK = FastPeakFind(MEAN_IMAGE*100,0.1);
[xx,yy] = meshgrid(1:size(MEAN_IMAGE,2),1:size(MEAN_IMAGE,1));
figure; [C1,~] = contour(xx,yy,MEAN_IMAGE-0.33); close;
[xC1,yC1,zC1] = C2xyz(C1); xC1(zC1~=0)=[]; yC1(zC1~=0)=[]; zC1(zC1~=0)=[];
figure; imagesc(MEAN_IMAGE); axis image xy; colormap(jet); colorbar;
hold on; plot(PEAK(1:2:end),PEAK(2:2:end),'w+','linewidth',1.);
for ii = 1:length(zC1)
    plot(cell2mat(xC1(ii)),cell2mat(yC1(ii)),'-w','linewidth',1.5);
end
title("Individual 2");