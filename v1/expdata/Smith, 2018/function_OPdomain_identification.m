function [Result] = function_OPdomain_identification(OP_MAP,SIG)
%% OP domain Identification
% OP domains: Local maxima points of local homogeneity index map
[xx,yy] = meshgrid(1:size(OP_MAP,2),1:size(OP_MAP,1));

% figure; hold on;
% imagesc(OP_MAP); title('Data OP map');
% colormap(hsv); caxis([-pi/2 pi/2]); axis image xy;
% c = colorbar; c.Label.String = 'Preferred orientation (radian)';

% Convert OP to unit vectors
OP_VMAP = exp(2i*OP_MAP);

% Iterate over each points, calculate local vector sum and assign LHI
OP_VMAP_reshape = reshape(OP_VMAP,[size(OP_MAP,1)*size(OP_MAP,2),1]);
xx_reshape = reshape(xx,[size(OP_MAP,1)*size(OP_MAP,2),1]);
yy_reshape = reshape(yy,[size(OP_MAP,1)*size(OP_MAP,2),1]);
LHI_reshape = zeros(size(OP_MAP,1)*size(OP_MAP,2),1);
for ii = 1:size(OP_VMAP_reshape,1)
    xx_temp = xx_reshape-xx_reshape(ii,1);
    yy_temp = yy_reshape-yy_reshape(ii,1);
    gauss_temp =  1/SIG^2/(2*pi)*exp(-(xx_temp.^2+yy_temp.^2)/SIG^2/2);
    LHI_reshape(ii) = abs(nansum(gauss_temp.*OP_VMAP_reshape));
    fprintf("Identifying OP domains... %2.1f%%\n",ii/size(OP_VMAP_reshape,1)*100);
end
LHI = reshape(LHI_reshape,size(OP_MAP));
OPD = FastPeakFind(LHI, 20);
figure;
a = subplot(121); imagesc(LHI); colormap(a, jet); colorbar; axis image; caxis([0 1]);
hold on; plot(OPD(1:2:end),OPD(2:2:end),'wo','LineWidth',2);
b = subplot(122); imagesc(OP_MAP); colormap(b, hsv); colorbar; axis image;
hold on; plot(OPD(1:2:end),OPD(2:2:end),'wo','LineWidth',2);

Result = [OPD(1:2:end)';OPD(2:2:end)'];

end