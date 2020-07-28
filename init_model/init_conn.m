function [w_2_1] = init_conn(pos1,pos2,r)
tic; disp("Initializing 1-to-2 connection");
n1 = size(pos1,1); n2 = size(pos2,1);
pos1_gpu = gpuArray(pos1);
pos2_gpu = gpuArray(pos2);

distx = repmat(pos2_gpu(:,1),1,n1)-repmat(pos1_gpu(:,1)',n2,1);
disty = repmat(pos2_gpu(:,2),1,n1)-repmat(pos1_gpu(:,2)',n2,1);
dist2 = distx.^2+disty.^2;

w_2_1 = gather(double(dist2<=r^2));
gpuDevice(1);
toc
end