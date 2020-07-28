function [Result] = init_feedforward(ff_w0_sig,pos_OFF,pos_ON,pos_V1,ff_w0_str,ff_w0_thr)
tic;
pos_OFF = gpuArray(pos_OFF);
pos_ON = gpuArray(pos_ON);
pos_V1 = gpuArray(pos_V1);

% V1<-ON
distx = pos_V1(:,1)-pos_ON(:,1)';
disty = pos_V1(:,2)-pos_ON(:,2)';

dist2 = distx.^2+disty.^2;
w0_V1_ON = exp(-1/2/ff_w0_sig^2*dist2);

dist = sqrt(dist2);
w0_V1_ON = exp(-dist/ff_w0_sig);

% V1<-OFF
distx = pos_V1(:,1)-pos_OFF(:,1)';
disty = pos_V1(:,2)-pos_OFF(:,2)';

dist2 = distx.^2+disty.^2;
w0_V1_OFF = exp(-1/2/ff_w0_sig^2*dist2);

dist = sqrt(dist2);
w0_V1_OFF = exp(-dist/ff_w0_sig);

% Thresholding
w0_V1_ON(w0_V1_ON<1e-04) = 0;
w0_V1_OFF(w0_V1_OFF<1e-04) = 0;

% Removing V1 cells with too low input
remove = (sum(w0_V1_ON,2)+sum(w0_V1_OFF,2))<ff_w0_thr;
pos_V1(remove,:) = [];
w0_V1_ON(remove,:) = [];
w0_V1_OFF(remove,:) = [];

% Normalize input weight
w0_V1_ON = w0_V1_ON./sum(w0_V1_ON,2)*ff_w0_str/2;
w0_V1_OFF = w0_V1_OFF./sum(w0_V1_OFF,2)*ff_w0_str/2;
w0_V1_ON(w0_V1_ON>1) = 1;
w0_V1_OFF(w0_V1_OFF>1) = 1;
w0_V1_ON(isnan(w0_V1_ON)) = 0;
w0_V1_OFF(isnan(w0_V1_OFF)) = 0;
Result = {gather(w0_V1_ON),gather(w0_V1_OFF),gather(pos_V1)};
gpuDevice(1); toc
end