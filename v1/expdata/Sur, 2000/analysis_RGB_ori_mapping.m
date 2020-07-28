function [OP_MAP] = analysis_RGB_ori_mapping(res,DATA_MAP,IF)
if IF
    %% (R,G,B) Curve for Orientation Mapping
    figure;
    subplot(2,2,1);
    image(DATA_MAP); title("Data"); axis image;
    subplot(2,2,2); imagesc(DATA_MAP(:,:,1)); title("R");
    axis image; colorbar; caxis([0 255]);
    subplot(2,2,3); imagesc(DATA_MAP(:,:,2)); title("G");
    axis image; colorbar; caxis([0 255]);
    subplot(2,2,4); imagesc(DATA_MAP(:,:,3)); title("B");
    axis image; colorbar; caxis([0 255]);
    
    RGB = zeros(3,res+1);
    ORI = (0:res)*pi/res;
    steps = 0:res;
    temp1 = (1-abs(steps/res-0)*3); temp2 = (1-abs(steps/res-1)*3);
    RGB(1,:) = temp1.*(temp1>0)+temp2.*(temp2>0);
    RGB(2,:) = (1-abs(steps/res-1/3)*3);
    RGB(2,:) = RGB(2,:).*(RGB(2,:)>0);
    RGB(3,:) = (1-abs(steps/res-2/3)*3);
    RGB(3,:) = RGB(3,:).*(RGB(3,:)>0);
    RGB = uint8(round(RGB*255));
    
    %% Orientation Mapping, Using Point-Curve Min Dist
    DATA_MAP_reshape = reshape(DATA_MAP, [size(DATA_MAP,1)*size(DATA_MAP,2) 3]);
    nearest_ori = zeros(size(DATA_MAP_reshape,1),1);
    for ii = 1:size(DATA_MAP_reshape,1)
        if sum(DATA_MAP_reshape(ii,:)) == 0
            nearest_ori(ii) = rand*pi;
        else
            min_dist = Inf;
            for jj = 1:res+1
                temp_dist = norm(double(DATA_MAP_reshape(ii,:)'-RGB(:,jj)));
                if min_dist > temp_dist
                    min_dist = temp_dist;
                    nearest_ori(ii) = ORI(jj);
                end
            end
            fprintf("%.3g percent complete... Current orientation %grad\n",ii/size(DATA_MAP_reshape,1)*100, nearest_ori(ii)/pi);
        end
    end
    OP_MAP = reshape(nearest_ori, [size(DATA_MAP,1) size(DATA_MAP,2)]);
else
    OP_MAP = load("OP_MAP.mat");
    OP_MAP = OP_MAP.OP_MAP;
end

end