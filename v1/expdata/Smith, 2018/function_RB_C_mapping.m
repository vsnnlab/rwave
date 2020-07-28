function [C_MAP] = function_RB_C_mapping(DATA_MAP,IF)
if IF
    %% (R,B) Curve for Orientation Mapping
    figure;
    subplot(2,2,1);
    image(DATA_MAP); title("Data"); axis image;
    subplot(2,2,2); imagesc(DATA_MAP(:,:,1)); title("R");
    axis image; colorbar; caxis([0 255]);
    subplot(2,2,3); imagesc(DATA_MAP(:,:,2)); title("G");
    axis image; colorbar; caxis([0 255]);
    subplot(2,2,4); imagesc(DATA_MAP(:,:,3)); title("B");
    axis image; colorbar; caxis([0 255]);
    
    res = 250;
    B = [ones(res,1)*255; linspace(255,1,res)'];
    G = [linspace(1,255,res)'; linspace(255,1,res)'];
    R = [linspace(1,255,res)';ones(res,1)*255];
    red_blue = [R G B];
    RB = uint8(red_blue);
    Val = linspace(-1,1,2*res);
    
    %% Orientation Mapping, Using Point-Curve Min Dist
    DATA_MAP_reshape = reshape(DATA_MAP, [size(DATA_MAP,1)*size(DATA_MAP,2) 3]);
    nearest = zeros(size(DATA_MAP_reshape,1),1);
    for ii = 1:size(DATA_MAP_reshape,1)
        if sum(DATA_MAP_reshape(ii,:)) == 0
            nearest(ii) = 1/0;
        else
            min_dist = Inf;
            for jj = 1:size(RB,1)
                temp_dist = norm(double(DATA_MAP_reshape(ii,:))-double(RB(jj,:)));
                if min_dist > temp_dist
                    min_dist = temp_dist;
                    nearest(ii) = jj;
                end
            end
            fprintf("%.3g percent complete... current index %g\n",ii/size(DATA_MAP_reshape,1)*100,nearest(ii));
        end
    end
    maxlim = size(Val,2); C_MAP = reshape(Val(min(nearest,maxlim)),[size(DATA_MAP,1) size(DATA_MAP,2)]);
    
    figure; scatter3(RB(:,1),RB(:,2),RB(:,3),10,double(RB)/255,'filled'); hold on;
    scatter3(DATA_MAP_reshape(:,1),DATA_MAP_reshape(:,2),DATA_MAP_reshape(:,3),10,Val(min(nearest,maxlim)),'filled');
    colormap(double(RB)/255); colorbar; caxis([-1 1]);
    
else
    C_MAP = load("C_MAP.mat");
    C_MAP = C_MAP.C_MAP;
end

end