close all; clear all;
%% OP Map Data Import & Preprocessing
% jpg: ? X ? X 3 uint8 format
DATA_OPMAP = imread('Data/RAW_OPMAP.png');
% RGB - OPMAP Conversion
res = 500;
OP_MAP = analysis_RGB_ori_mapping(res,DATA_OPMAP,true);
% Gaussian Noise Reduction
SIG = 10; OP_MAP_F = angle(imgaussfilt(real(exp(2i*OP_MAP)),SIG)+1i*imgaussfilt(imag(exp(2i*OP_MAP)),SIG))/2;

OP_MAP_F(OP_MAP_F<0) = OP_MAP_F(OP_MAP_F<0)+pi;
OP_MAP = -OP_MAP_F+pi/6;
OP_MAP(OP_MAP>pi) = OP_MAP(OP_MAP>pi)-pi;

NEW_OP_MAP = OP_MAP;

figure;
subplot(221); image(DATA_OPMAP); axis image; title("RGB data image");
subplot(223); imagesc(OP_MAP_F); axis image; colormap(hsv); colorbar; title("Orientation image");
subplot(224); imagesc(OP_MAP); axis image; colormap(hsv); colorbar; title("Filtered OP image");
LEN = size(OP_MAP,1)*size(OP_MAP,2);
figure; scatter3(...
    reshape(DATA_OPMAP(:,:,1),1,LEN),...
    reshape(DATA_OPMAP(:,:,2),1,LEN),...
    reshape(DATA_OPMAP(:,:,3),1,LEN),...
    10,reshape(OP_MAP,1,LEN),'filled'); colormap(hsv);
hold on; scatter3(RGB(1,:),RGB(2,:),RGB(3,:),10,ORI,'filled');

OP_MAP = OP_MAP_F;
% Bouton detection
DATA_OPMAP_reshape = double(reshape(DATA_OPMAP, [size(DATA_OPMAP,1)*size(DATA_OPMAP,2) 3]));
Bouton = imgaussfilt(double(reshape(sum(DATA_OPMAP_reshape,2)<225,size(DATA_OPMAP,1),size(DATA_OPMAP,2))),1);
% gcf = figure;
% a = subplot(131); imagesc(OP_MAP); axis image; colormap(a,hsv); colorbar; title("Orientation image");
% b = subplot(132); imagesc(Bouton>0.1); title("Bouton"); colormap(b,flipud(gray)); axis image; colorbar;
% c = subplot(133);
Injection = [148 152];
DIFF = OP_MAP-OP_MAP(148,152); DIFF = abs(DIFF); DIFF(DIFF>pi/2) = pi-DIFF(DIFF>pi/2);
OP_scaled = (OP_MAP-min(OP_MAP(:)))/(max(OP_MAP(:))-min(OP_MAP(:)));
rgbImage_DIFF = ind2rgb(round(DIFF/max(max(DIFF))*255), gray(255));
rgbImage_OPMAP = ind2rgb(round(OP_scaled*255), hsv(255));
rgbImage_bouton = ind2rgb(round(Bouton/max(max(Bouton))*255), jet(255));
rgbImage_bouton_ox = ind2rgb((Bouton<0.1)*255, gray(255));
%% Plot import and merge
figure(777); set(gcf, 'Position', [50 50 1300 200]);
subplot(141);
% image(rgbImage_DIFF); axis image off; hold on; colormap(gray); colorbar('Ticks',[0 1],'TickLabels',[0 pi/2]);
% I = image(rgbImage_bouton,'alphadata',1*(Bouton>0.1));
% title("Orientation difference"); set(gcf,"Position",[50 150 1800 400]);
% plot(148,152,"go","linewidth",3,"Markersize",5);
subplot(142);
dist1 = load("Data/Dist1.csv"); [dist,ind] = sort(dist1(:,1)); number = abs(dist1(ind,2));
plot(dist,number,'k','linewidth',3); pbaspect([1 1 1]); xlim([0.5 5]); ylim([0 1800]);
xlabel("Distance from injection site (mm)");
xticks([0.5 1 2 3 4 5]); xticklabels(["0.5" "1.0" "2.0" "3.0" "4.0" "5.0"]);
yticks([0 300 600 900 1200 1500 1800]); ylabel("Number of boutons");
title("Bouton distribution along preferred axis");
subplot(143);
SCT = load("Data/Scatter.csv"); scatter(SCT(:,1),SCT(:,2),'dk',"linewidth",2);
xlim([0 4]); ylim([0 4]); axis image; hold on; plot([0 4],[0 4],'--k');
xlabel("Max distance preferred axis (mm)"); ylabel("Max distance orthogonal axis (mm)");
title("Long-ranged bouton distributions");
subplot(144);
polar = load("Data/Polar1.csv"); [theta,ind] = sort(polar(:,2)); theta = theta*pi/180; rho = polar(ind,1);
theta = [theta; theta(1)]; rho = [rho; rho(1)];
polarplot(theta,rho,'k','linewidth',3); rticks([]); rlim([0 max(rho)]);
thetaticklabels(string(0:30:330)+char(176));
hold on; polarplot(0,0,'+k','markersize',10,'linewidth',3); polarplot([0 pi],[max(rho) max(rho)],'--k');
%legend("Median (13 cases)");
title("Orientation specificity of bouton distributions");