function [mean1,mean2,std1,std2,p] = analysis_CORR_OPSIM_significance(CORR,OP_sim,N)
Valid_Pearson = zeros(2,N);
Random_Pearson = zeros(2,N);
exclude = zeros(1,N);
xsize = size(OP_sim,2);
ysize = size(OP_sim,1);
for nn = 1:N
    % Random image: random rotation and translation
    theta = rand*360;
    %dx = round((rand*2-1)/2*xsize);
    %dy = round((rand*2-1)/2*ysize);
    rotsize = ceil(sqrt(xsize^2+ysize.^2));
    %dmax = max(abs(dx),abs(dy));
    dx = 0; dy = 0; dmax = 0;
    Corr_rot = function_imrotate(CORR,theta);
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
    Corr_original_1d = CORR(:);
    Corr_original_overlap = Corr_original_1d(overlap_1d);
    OP_original_1d = OP_sim(:);
    OP_overlap = OP_original_1d(overlap_1d);
    if N==1
        [R,P] = corrcoef(Corr_original_1d,OP_original_1d);
        Valid_Pearson(:,nn) = [R(1,2);P(1,2)];
    else
        [R,P] = corrcoef(Corr_original_overlap,OP_overlap);
        Valid_Pearson(:,nn) = [R(1,2);P(1,2)];
    end
    [R,P] = corrcoef(Corr_rand_overlap,OP_overlap);
    Random_Pearson(:,nn) = [R(1,2);P(1,2)];
end
% Paired t-test
if N~=1
    [h,p] = ttest(Valid_Pearson(1,:)',Random_Pearson(1,:)');
    mean1 = mean(Valid_Pearson(1,~exclude));
    mean2 = mean(Random_Pearson(1,~exclude));
    std1 = std(Valid_Pearson(1,~exclude));
    std2 = std(Random_Pearson(1,~exclude));
else
    mean1 = Valid_Pearson(1,1);
    mean2 = Random_Pearson(1,1);
    std1 = 0;
    std2 = 0;
    p = 0;
end
% figure; errorbar([1,2],[mean(Valid_Pearson(1,~exclude)),mean(Random_Pearson(1,~exclude))],...
%     [std(Valid_Pearson(1,~exclude)),std(Random_Pearson(1,~exclude))]); ylim([-0.2 0.8]); xlim([0.5 2.5]);
% suptitle(num2str(p));
% disp(p);



% figure;
% a = subplot(141); imagesc(OP_sim); axis image; colormap(a,flipud(gray)); colorbar('southoutside'); caxis([0 1]);
% b = subplot(142); imagesc(CORR); axis image; colormap(b,redblue); colorbar('southoutside'); caxis([-0.75 0.75]);
% c = subplot(143); imagesc(Corr_rand); axis image; colormap(c,redblue); colorbar('southoutside'); caxis([-0.75 0.75]);
% d = subplot(144); imagesc(overlap); axis image; colormap(d,redblue); colorbar('southoutside'); caxis([-0.75 0.75]);
end