function [Result] = RGC_AC_mosaic(crop_ON,crop_OFF,pad_r,crop_x,crop_y)
disp("Generate RGC padding and AC mosaics");
% Lattice spacing
density_ON = size(crop_ON,1)/4/crop_x/crop_y;
density_OFF = size(crop_OFF,1)/4/crop_x/crop_y;
d_ON = sqrt(2/sqrt(3)/density_ON);
d_OFF = sqrt(2/sqrt(3)/density_OFF);
d_AC = sqrt(2/sqrt(3)/(density_ON+density_OFF));
theta = rand*pi; rot_OFF = [cos(theta) sin(theta); -sin(theta) cos(theta)];
theta = rand*pi; rot_AC = [cos(theta) sin(theta); -sin(theta) cos(theta)];

% RGC mosaic
hex = 1/2*[1 1;sqrt(3) -sqrt(3)];
ij = combvec(-100:100,-100:100)';
Lij = ij*hex';
pos_ON = d_ON*Lij;
pos_OFF = d_OFF*Lij*rot_OFF';
pos_AC = d_AC*Lij*rot_AC';

% Give displacement
pos_OFF = pos_OFF+[500 1250];
pos_AC = pos_AC+[170 310];

% Restrict
pos_ON(pos_ON(:,1).^2+pos_ON(:,2).^2>pad_r^2,:) = [];
pos_OFF(pos_OFF(:,1).^2+pos_OFF(:,2).^2>pad_r^2,:) = [];
pos_AC(pos_AC(:,1).^2+pos_AC(:,2).^2>pad_r^2,:) = [];
pos_ON(abs(pos_ON(:,1))<crop_x&abs(pos_ON(:,2))<crop_y,:) = [];
pos_OFF(abs(pos_OFF(:,1))<crop_x&abs(pos_OFF(:,2))<crop_y,:) = [];

Result = {pos_ON, pos_OFF, pos_AC, d_ON, d_OFF};
end