%% OP Map Data Import & Preprocessing
% jpg: ? X ? X 3 uint8 format
DATA_OPMAP = imread('Data/RAW_OPMAP2.jpg');
% White Space Filling (using 0)
DATA_OPMAP_reshape = reshape(DATA_OPMAP, [size(DATA_OPMAP,1)*size(DATA_OPMAP,2) 3]);
thr = 190;
white_mask = (DATA_OPMAP_reshape(:,1)>thr) & (DATA_OPMAP_reshape(:,2)>thr) & (DATA_OPMAP_reshape(:,3)>thr);
DATA_OPMAP_reshape(white_mask,1) = 0;
DATA_OPMAP_reshape(white_mask,2) = 0;
DATA_OPMAP_reshape(white_mask,3) = 0;
DATA_OPMAP = reshape(DATA_OPMAP_reshape,[size(DATA_OPMAP,1),size(DATA_OPMAP,2) 3]);
% RGB - OPMAP Conversion
res = 500;
OP_MAP = function_RGB_OP_mapping(res,DATA_OPMAP,true);
OP_MAP(isinf(OP_MAP)) = nan;
% Gaussian Noise Reduction
MASK = ~isnan(OP_MAP);
OP_MAP_F = fillmissing(OP_MAP,'nearest');
OP_MAP_F = angle(imgaussfilt(real(exp(2i*OP_MAP_F)),1.5,'FilterDomain','spatial')+...
    1i*imgaussfilt(imag(exp(2i*OP_MAP_F)),1.5,'FilterDomain','spatial'))/2;
OP_MAP_F = OP_MAP_F.*MASK;
OP_MAP_F(~MASK) = nan;
NEW_OP_MAP = OP_MAP_F;
OP_MAP_F(OP_MAP_F<0) = OP_MAP_F(OP_MAP_F<0)+pi;
% figure;
figure;
subplot(1,2,1); image(DATA_OPMAP); axis image; title("RGB data image");
subplot(1,2,2); imagesc(OP_MAP_F); hold on; colormap(hsv); axis image;
c = colorbar; c.Label.String = 'Preferred orientation (radian)';
%% Digitize correlation pattern
% jpg: ? X ? X 3 uint8 format
DATA_CMAP = imread('Data/RAW_CORR2.jpg');
% RGB - OPMAP Conversion
C_MAP = function_RB_C_mapping(DATA_CMAP,true)*0.75;
%% Contour plot
OP_point = [73 192];
DIFF = abs(OP_MAP_F-OP_MAP_F(OP_point(2),OP_point(1))); DIFF(DIFF>pi/2) = pi-DIFF(DIFF>pi/2);
figure; imagesc(NEW_OP_MAP); axis image;
[xx,yy] = meshgrid(1:size(NEW_OP_MAP,2),1:size(NEW_OP_MAP,1));
[C1_OP,~] = contour(xx,yy,DIFF-pi/4); close;
[xC1_OP,yC1_OP,zC1_OP] = C2xyz(C1_OP);
xC1_OP(zC1_OP~=0)=[]; yC1_OP(zC1_OP~=0)=[]; zC1_OP(zC1_OP~=0)=[];
figure; a = subplot(121); imagesc(OP_MAP_F); title('Orientation map'); colormap(a,hsv); axis image; hold on;
plot(73,192,"og","LineWidth",3);
for ii = 1:length(zC1_OP)
    plot(cell2mat(xC1_OP(ii)),cell2mat(yC1_OP(ii)),'-k','linewidth',1.5);
end
b = subplot(122); imagesc(C_MAP); axis image; colormap(b,redblue); caxis([-0.75 0.75]); hold on;
plot(67,192,"og","LineWidth",3); plot(275,192,"og","LineWidth",3);
plot(483,192,"og","LineWidth",3); plot(691,192,"og","LineWidth",3);
for ii = 1:length(zC1_OP)
    plot(cell2mat(xC1_OP(ii))-73+67,cell2mat(yC1_OP(ii)),'-k','linewidth',1.5);
    plot(cell2mat(xC1_OP(ii))-73+275,cell2mat(yC1_OP(ii)),'-k','linewidth',1.5);
    plot(cell2mat(xC1_OP(ii))-73+483,cell2mat(yC1_OP(ii)),'-k','linewidth',1.5);
    plot(cell2mat(xC1_OP(ii))-73+691,cell2mat(yC1_OP(ii)),'-k','linewidth',1.5);
end
set(gcf,"Position",[50 150 1000 400]);
%% OP-corrpattern matching
Seed1 = [67 192];
Seed2 = [275 192];
Seed3 = [483 192];
Seed4 = [691 192];

OP_MAP_CUT = OP_MAP_F(:,OP_point(1)-65:OP_point(1)+80);
DIFF = DIFF(:,OP_point(1)-65:OP_point(1)+80);
OP_sim = (pi/2-DIFF)/max(DIFF(:)); res_OP_sim = OP_sim(:); MASK = ~isnan(res_OP_sim);
[xx,yy] = meshgrid(1:size(OP_sim,2),1:size(OP_sim,1)); [C1_OP,~] = contour(xx,yy,DIFF-pi/4); close;
[xC1_OP,yC1_OP,zC1_OP] = C2xyz(C1_OP); xC1_OP(zC1_OP~=0)=[]; yC1_OP(zC1_OP~=0)=[]; zC1_OP(zC1_OP~=0)=[];

Corr1 = C_MAP(:,Seed1(1)-65:Seed1(1)+80); res_Corr1 = Corr1(:);
Corr2 = C_MAP(:,Seed2(1)-65:Seed2(1)+80); res_Corr2 = Corr2(:);
Corr3 = C_MAP(:,Seed3(1)-65:Seed3(1)+80); res_Corr3 = Corr3(:);
[xx,yy] = meshgrid(1:size(Corr3,2),1:size(Corr3,1)); [C3,~] = contour(xx,yy,Corr3); close;
[xC3,yC3,zC3] = C2xyz(C3); xC3(zC3~=0)=[]; yC3(zC3~=0)=[]; zC3(zC3~=0)=[];
Corr4 = C_MAP(:,Seed4(1)-65:Seed4(1)+80); res_Corr4 = Corr4(:);

figure; suptitle("Pearson corrcoef / p-value");
a = subplot(141); imagesc(Corr3); axis image; colormap(a,redblue); colorbar('southoutside'); caxis([-0.75 0.75]);
[R,P] = corrcoef(res_OP_sim(MASK),res_Corr3(MASK));
title("N = "+num2str(sum(MASK))+", r = "+num2str(R(1,2))+", p = "+num2str(P(1,2)));
hold on; plot(65,192,"og","LineWidth",3);
for ii = 1:length(zC3)
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
end
b = subplot(142); imagesc(OP_MAP_CUT); axis image; colormap(b,hsv); colorbar('southoutside'); hold on;
for ii = 1:length(zC1_OP)
    plot(cell2mat(xC1_OP(ii)),cell2mat(yC1_OP(ii)),'-k','linewidth',1.5);
end
c = subplot(143); imagesc(OP_sim); axis image; colormap(c,gray); colorbar('southoutside'); caxis([0 1]);
title("Scaled OP similarity"); hold on;
for ii = 1:length(zC1_OP)
    plot(cell2mat(xC1_OP(ii)),cell2mat(yC1_OP(ii)),'-k','linewidth',1.5);
end
for ii = 1:length(zC3)
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
    plot(cell2mat(xC3(ii)),cell2mat(yC3(ii)),'-g','linewidth',1.5);
end
% Statistical significance
OP_sim_temp = OP_sim(60:200,40:120);
Corr3_temp = Corr3(60:200,40:120);
Valid_Pearson = zeros(2,100);
Random_Pearson = zeros(2,100);
exclude = zeros(1,100);
xsize = size(OP_sim_temp,2);
ysize = size(OP_sim_temp,1);
for nn = 1:100
    % Random image: random rotation and translation
    theta = rand*360;
    dx = round((rand*2-1)/2*xsize);
    dy = round((rand*2-1)/2*ysize);
    dmax = max(abs(dx),abs(dy));
    dx = 0; dy = 0; dmax = 0;
    rotsize = ceil(sqrt(xsize^2+ysize.^2));
    Corr_rot = function_imrotate(Corr3_temp,theta);
    Corr_trans = zeros(2*rotsize+dmax,2*rotsize+dmax)*nan;
    xcen = round(size(Corr_trans,2)/2)+dx;
    ycen = round(size(Corr_trans,1)/2)+dy;
    xfill = round(xcen-size(Corr_rot,2)/2)+1:round(xcen+size(Corr_rot,2)/2);
    yfill = round(ycen-size(Corr_rot,1)/2)+1:round(ycen+size(Corr_rot,1)/2);
    Corr_trans(yfill,xfill) = Corr_rot;
    % Crop overlapping region
    ly = max(round(rotsize+dmax-ysize/2)+1,1);
    Uy = min(round(rotsize+dmax+ysize/2),size(Corr_trans,2));
    lx = max(round(rotsize+dmax-xsize/2)+1,1);
    Ux = min(round(rotsize+dmax+xsize/2),size(Corr_trans,1));
    Corr_rand = Corr_trans(ly:Uy,lx:Ux);
    Corr_rand = imresize(Corr_rand,ysize/size(Corr_rand,1));
    % Compute correlation
    overlap = ~isnan(Corr_rand);
    if sum(overlap(:)) <= 10; disp("<= 10 pixel overlap!"); exclude(nn) = true; continue; end
    overlap_1d = overlap(:);
    Corr_rand_1d = Corr_rand(:);
    Corr_rand_overlap = Corr_rand_1d(overlap_1d);
    Corr_original_1d = Corr3_temp(:);
    Corr_original_overlap = Corr_original_1d(overlap_1d);
    OP_original_1d = OP_sim_temp(:);
    OP_overlap = OP_original_1d(overlap_1d);
    [R,P] = corrcoef(Corr_original_overlap,OP_overlap);
    Valid_Pearson(:,nn) = [R(1,2);P(1,2)];
    [R,P] = corrcoef(Corr_rand_overlap,OP_overlap);
    Random_Pearson(:,nn) = [R(1,2);P(1,2)];
end
% Paired t-test
[h,p] = ttest(Valid_Pearson(1,:)',Random_Pearson(1,:)'); disp(h);
subplot(144); errorbar([1,2],[mean(Valid_Pearson(1,~exclude)),mean(Random_Pearson(1,~exclude))],...
    [std(Valid_Pearson(1,~exclude)),std(Random_Pearson(1,~exclude))]); ylim([-0.4 1]); xlim([0.5 2.5]);
title(sprintf("Activity/orientation match\nN = "+num2str(size(Valid_Pearson,2))+...
    ", paired t-test p = "+num2str(p)));
%%
hold on; plot(65,192,"og","LineWidth",3);
b = subplot(152); imagesc(Corr1); axis image; colormap(b,redblue); colorbar('southoutside'); caxis([-0.75 0.75]);
[R,P] = corrcoef(res_OP_sim(MASK),res_Corr1(MASK)); title(num2str(R(1,2))+" / "+num2str(P(1,2)));
hold on; plot(65,192,"og","LineWidth",3);
c = subplot(153); imagesc(Corr2); axis image; colormap(c,redblue); colorbar('southoutside'); caxis([-0.75 0.75]);
[R,P] = corrcoef(res_OP_sim(MASK),res_Corr2(MASK)); title(num2str(R(1,2))+" / "+num2str(P(1,2)));
hold on; plot(65,192,"og","LineWidth",3);
d = subplot(154); imagesc(Corr3); axis image; colormap(d,redblue); colorbar('southoutside'); caxis([-0.75 0.75]);
[R,P] = corrcoef(res_OP_sim(MASK),res_Corr3(MASK)); title(num2str(R(1,2))+" / "+num2str(P(1,2)));
hold on; plot(65,192,"og","LineWidth",3);


SHOW = randsample(1:length(res_OP_sim(MASK)),100);
OPsim = res_OP_sim(MASK); CORR4 = res_Corr4(MASK);
figure; scatter(OPsim,CORR4); hold on;
l = lsline ; set(l,'lineWidth',2);
xlim([0 1]); ylim([-1 1]);
