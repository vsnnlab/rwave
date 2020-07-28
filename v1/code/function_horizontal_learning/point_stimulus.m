function [V1_point_input] = point_stimulus(pos_V1,sig,n_events)
tic; disp("Point stimulus generation");
n_V1 = size(pos_V1,1);
V1_point_input = gpuArray(zeros(n_V1,n_events));
temp = gpuArray(zeros(n_V1,n_events));
xmin = min(pos_V1(:,1));
xmax = max(pos_V1(:,1));
ymin = min(pos_V1(:,2));
ymax = max(pos_V1(:,2));
margin = 100;
for event = 1:n_events
    % Cortical stimulus generation
    loc = [xmin+margin+rand*(xmax-xmin-2*margin), ymin+margin+rand*(ymax-ymin-2*margin)];
    temp(:,event) = exp(-vecnorm(pos_V1-loc,2,2).^2/sig^2/2); %double(vecnorm(pos_V1-loc,2,2)<sig);
    V1_point_input(:,event) = temp(:,event)/max(temp(:,event));
end
V1_point_input = gather(V1_point_input); gpuDevice(1); toc
end