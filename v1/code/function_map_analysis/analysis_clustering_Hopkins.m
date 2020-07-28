function [CI] = analysis_clustering_Hopkins(pos_V1,w_V1_V1,d_OFF,crop_x,crop_y)
% Hopkins' statistics for evaluating clusters
% Peak sensitivity (sliding window size) set to 1 cycle / lambda
n_V1 = size(w_V1_V1,1);
w_V1_V1 = gpuArray(w_V1_V1);
n_syn = round(n_V1*1/5);
roi_size = 2*d_OFF;
roi_slide_x = linspace(-crop_x+roi_size,crop_x-roi_size,11);
roi_slide_y = linspace(-crop_y+roi_size,crop_y-roi_size,11);

roi_CI = zeros(1,size(roi_slide_x,2));
CI = [];

presynaptic = randperm(n_V1);
cnt = 0;
while cnt<50
    ii = presynaptic(1);
    presynaptic(1) = [];
    % Select output synapse locations
    [~,ind_V1] = sort(w_V1_V1(:,ii),'descend');
    connect_V1 = pos_V1(ind_V1(1:n_syn),:);
    refcell_CI = [];
    % Slide ROI window and compute values for Hopkins' statistics
    for xcen = roi_slide_x
        for ycen = roi_slide_y
            if sum(([xcen ycen]-pos_V1(ii,:)).^2,2)<(200+roi_size)^2
                roi_CI = nan;
                refcell_CI = [refcell_CI mean(roi_CI)];
                continue
            end
            % V1 cells in ROI
            [~,ind_roi] = find(((connect_V1(:,1)'-xcen).^2+(connect_V1(:,2)'-ycen).^2)<(roi_size)^2);
            roi_V1 = connect_V1(ind_roi,:);
            n = size(roi_V1,1); if n<10; continue; end; m = ceil(n/10); x = zeros(1,m); w = zeros(1,m);
            rec_pos = [xcen ycen];
            for jj = 1:10
                randsample_V1 = roi_V1(randsample(n,m),:); % Generate random 10%-subset
                a = rand(m,1)*2*pi;
                r = roi_size*sqrt(rand(m,1));
                xrand = r.*cos(a)+xcen;
                yrand = r.*sin(a)+ycen;
                randpos_V1 = [xrand yrand]; % Generate random 10%-positions
                % Determine H-statistics
                for kk = 1:m
                    % x: nearest distances to whole ROI set, from random positions
                    % w: nearest distances to whole ROI set, from random subset
                    temp_x = min(sum((roi_V1-randpos_V1(kk,:)).^2,2)); x(kk) = sqrt(temp_x);
                    temp_w = sort(sum((roi_V1-randsample_V1(kk,:)).^2,2),'ascend'); w(kk) = sqrt(temp_w(2));
                end
                roi_CI(jj) = log(sum(x.^2)/sum(w.^2));
            end
            refcell_CI = [refcell_CI mean(roi_CI)];
        end
    end
    
    if ~isnan(median(refcell_CI,'omitnan'))
        cnt = cnt+1;
        CI = [CI median(refcell_CI,'omitnan')];
        disp(CI(length(CI)))
    end
end

% figure;
% scatter(pos_V1(:,1),pos_V1(:,2),10,'r','filled'); axis xy image; hold on;
% scatter(pos_V1(ii,1),pos_V1(ii,2),20,'g','filled');
% scatter(connect_V1(:,1),connect_V1(:,2),20,'b','filled');
% scatter(roi_V1(:,1),roi_V1(:,2),30,'k','filled');
% plot(randpos_V1(:,1),randpos_V1(:,2),'rd');
% plot(randsample_V1(:,1),randsample_V1(:,2),'ro');
% viscircles([rec_pos(1) rec_pos(2)], roi_size);
end