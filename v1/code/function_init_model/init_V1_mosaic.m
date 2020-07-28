function [pos_V1] = init_V1_mosaic(d_V1,d_OFF,crop_x,crop_y,pos_ON,pos_OFF,type)
tic; pos_V1 = [];
if type == 1
    % Regular hexagonal mosaic
    theta = rand*pi; rot = [cos(theta) sin(theta); -sin(theta) cos(theta)];
    hex = 1/2*[1 1;sqrt(3) -sqrt(3)];
    ij = combvec(-100:100,-100:100)';
    Lij = ij*hex';
    pos_V1 = d_V1*Lij*rot';
    noise = randn(size(pos_V1))*d_V1*0.1; % std = 10% d_V1
    pos_V1 = pos_V1 + noise;
end

if type == 2
    % Dipole sampling
    for ii = 1:size(pos_OFF,1)
        temp_OFF = pos_OFF(ii,:);
        dist2 = sum((pos_ON-temp_OFF).^2,2);
        near_ONs = find(dist2<(d_OFF*1.5)^2);
        pos_V1 = [pos_V1;(pos_ON(near_ONs,:)+temp_OFF)/2];
    end
end

if type == 3
    % Nearest dipole
    for ii = 1:size(pos_OFF,1)
        temp_OFF = pos_OFF(ii,:);
        dist2 = sum((pos_ON-temp_OFF).^2,2);
        [~,near_ON] = min(dist2);
        pos_V1 = [pos_V1;(pos_ON(near_ON,:)+temp_OFF)/2];
    end
end

% Restrict
pos_V1(pos_V1(:,1)<-crop_x,:) = []; pos_V1(pos_V1(:,1)>crop_x,:) = [];
pos_V1(pos_V1(:,2)<-crop_y,:) = []; pos_V1(pos_V1(:,2)>crop_y,:) = [];
gpuDevice(1); toc
end