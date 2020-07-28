function [MEAN_IMAGE] = function_autocorrelation_analysis(num_ROI,X0,Y0,ROI_SIZE,C_MAP,restrict)
RES = zeros(100,100,num_ROI);
for ii = 1:num_ROI
    x0 = X0(ii); y0 = Y0(ii); dx = ROI_SIZE; dy = ROI_SIZE;
%     figure; imagesc(C_MAP); axis image xy; colormap(redblue); colorbar; caxis([-0.75 0.75]);
%     hold on; plot([x0 x0 x0+dx x0+dx],[y0 y0+dy y0 y0+dy],'ko');
    % Autocorrelation
    ROI = C_MAP(y0:y0+dy,x0:x0+dx); AUTOCORR = gather(xcorr2(gpuArray(ROI)));
    CENTER = [size(AUTOCORR,2);size(AUTOCORR,1)]/2;
    figure;
    a = subplot(221); imagesc(ROI); axis image xy; colormap(a,redblue); colorbar; caxis([-0.75 0.75]);
    b = subplot(222); imagesc(AUTOCORR); axis image xy; colormap(b,jet); colorbar; hold on; 
    PEAK = FastPeakFind(AUTOCORR/max(AUTOCORR(:))*100);
    PX = PEAK(1:2:end); PY = PEAK(2:2:end);
    PEAKDIST = sqrt((PX-CENTER(1)).^2+(PY-CENTER(2)).^2);
    PX(PEAKDIST>size(AUTOCORR,1)/restrict) = [];
    PY(PEAKDIST>size(AUTOCORR,1)/restrict) = [];
    PEAKDIST = sqrt((PX-CENTER(1)).^2+(PY-CENTER(2)).^2);
    PX(PEAKDIST<size(AUTOCORR,1)/10) = [];
    PY(PEAKDIST<size(AUTOCORR,1)/10) = [];
    P_STR = zeros(size(PX));
    for pp = 1:size(PX,1); P_STR(pp) = AUTOCORR(PY(pp),PX(pp)); end
    [~,pind] = sort(P_STR); S_IND = pind(end); SECONDARY = [PX(S_IND);PY(S_IND)];plot(PX,PY,'r+');
    plot(CENTER(1),CENTER(2),'wd'); plot(PX(S_IND),PY(S_IND),'kd');
    title("Mexican hat model, ROI "+num2str(ii));
    theta = atan((PX(S_IND)-CENTER(1))/(PY(S_IND)-CENTER(2)));
    % Create rotation matrix
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    % Rotate your point(s)
    ROTATED = imrotate(AUTOCORR,-theta*180/pi,'bilinear');
    R_CENTER = [size(ROTATED,2);size(ROTATED,1)]/2;
    R_SECONDARY = R*(SECONDARY-CENTER)+R_CENTER;
    c = subplot(223); imagesc(ROTATED); axis image xy; colormap(c,jet); colorbar; hold on;
    plot(R_CENTER(1),R_CENTER(2),'wd'); plot(R_SECONDARY(1),R_SECONDARY(2),'kd');
    WINSIZE = sqrt(sum((R_CENTER-R_SECONDARY).^2,1))*1.5;
    CUT = ROTATED(R_CENTER(2)-WINSIZE:R_CENTER(2)+WINSIZE,R_CENTER(1)-WINSIZE:R_CENTER(1)+WINSIZE);
    TEMP = imresize(CUT,100/size(CUT,1)); TEMP = imresize(TEMP,100/size(TEMP,1)); TEMP = TEMP/max(max(TEMP));
    d = subplot(224); imagesc(TEMP); axis image xy; colormap(d,jet); colorbar;
    RES(:,:,ii) = TEMP; RES_CENTER = [size(TEMP,2);size(TEMP,1)]/2;
end
MEAN_IMAGE = mean(RES,3); MEAN_IMAGE = MEAN_IMAGE/max(max(MEAN_IMAGE));
end