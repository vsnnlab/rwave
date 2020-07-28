function [Result] = function_pinwheel_identification(OP_MAP,winsize)
%% Pinwheel Identification
% Pinwheels: point of intersection between real and imag zero-magnitude contour of complex orientation map
[xx,yy] = meshgrid(1:size(OP_MAP,2),1:size(OP_MAP,1));

% figure; hold on;
% imagesc(OP_MAP); title('Data OP map');
% colormap(hsv); caxis([-pi/2 pi/2]); axis image xy;
% c = colorbar; c.Label.String = 'Preferred orientation (radian)';

OP_RMAP = real(exp(2i*OP_MAP));
OP_IMAP = imag(exp(2i*OP_MAP));
[C1,~] = contour(xx,yy,OP_RMAP);
[C2,~] = contour(xx,yy,OP_IMAP);
axis xy image off; close;

[x1,y1,z1] = C2xyz(C1);
[x2,y2,z2] = C2xyz(C2);

x1(z1~=0)=[]; y1(z1~=0)=[]; z1(z1~=0)=[];
x2(z2~=0)=[]; y2(z2~=0)=[]; z2(z2~=0)=[];
Pwl = []; Pwl_p = []; Pwl_n = [];
for ii = 1:length(z1)
    x1temp = cell2mat(x1(ii));
    y1temp = cell2mat(y1(ii));
    for radial_ind = 1:length(z2)
        x2temp = cell2mat(x2(radial_ind));
        y2temp = cell2mat(y2(radial_ind));
        Ptemp = round(InterX([x1temp;y1temp],[x2temp;y2temp]));
        Pwl = [Pwl Ptemp];
    end
end
for Ptemp = Pwl
    img_temp = OP_MAP(max(1,Ptemp(2)-winsize):min(size(OP_MAP,1),Ptemp(2)+winsize),...
        max(1,Ptemp(1)-winsize):min(size(OP_MAP,2),Ptemp(1)+winsize));
    cnt = 0;
    for kk = 1:size(img_temp,1)-1
        if img_temp(kk+1,1)-img_temp(kk,1)>=0
            cnt = cnt+1;
        else
            cnt = cnt-1;
        end
    end
    for kk = 1:size(img_temp,2)-1
        if img_temp(end,kk+1)-img_temp(end,kk)>=0
            cnt = cnt+1;
        else
            cnt = cnt-1;
        end
    end
    for kk = size(img_temp,1):1
        if img_temp(kk-1,end)-img_temp(kk,end)>=0
            cnt = cnt+1;
        else
            cnt = cnt-1;
        end
    end
    for kk = size(img_temp,2):1
        if img_temp(1,kk-1)-img_temp(1,kk)>=0
            cnt = cnt+1;
        else
            cnt = cnt-1;
        end
    end
    if cnt>0
        Pwl_p = [Pwl_p Ptemp];
    else
        Pwl_n = [Pwl_n Ptemp];
    end
end

Result = {Pwl_p, Pwl_n};
end