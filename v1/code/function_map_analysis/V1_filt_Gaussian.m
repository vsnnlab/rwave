function [map] = V1_filt_Gaussian(crop_x,crop_y,imgsize_x,img_sig,pos_V1,vals,circ)
imgsize_y = round(imgsize_x*crop_y/crop_x);

pos_V1(:,1) = pos_V1(:,1)*imgsize_x/2/crop_x+imgsize_x/2;
pos_V1(:,2) = pos_V1(:,2)*imgsize_y/2/crop_y+imgsize_y/2;
n_V1 = size(pos_V1,1);
[xx,yy] = meshgrid(1:imgsize_x,1:imgsize_y);

distx = xx-reshape(pos_V1(:,1),[1 1 n_V1]);
disty = yy-reshape(pos_V1(:,2),[1 1 n_V1]);
gauss = exp(-(distx.^2+disty.^2)/img_sig^2/2);

if circ
    map = gauss.*exp(2i*reshape(vals,[1 1 n_V1]));
    map = nansum(map,3);
    map = angle(map)/2;
    map(map<-pi/2) = map(map<-pi/2)+pi;
    map(map>pi/2) = map(map>pi/2)-pi;
else
    minv = min(vals); maxv = max(vals);
    norm_vals = (vals-minv)/(maxv-minv);
    map = gauss.*exp(1i*reshape(norm_vals,[1 1 n_V1]));
    map = nansum(map,3);
    map = angle(map);
    map = map*(maxv-minv)+minv;
end
end