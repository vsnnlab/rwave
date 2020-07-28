function [V1_noise_input] = background_noise_stimulus(pos_V1,sig,n_events)
tic; disp("Background noise generation");
n_V1 = size(pos_V1,1);
V1_noise_input = abs(gpuArray.randn(n_V1,n_events));
xdist = abs(pos_V1(:,1)-pos_V1(:,1)');
ydist = abs(pos_V1(:,2)-pos_V1(:,2)');
dist2 = xdist.^2+ydist.^2;
dist_weight = exp(-dist2/sig^2/2);
dist_weight = dist_weight./sum(dist_weight,2);
V1_noise_input = dist_weight*V1_noise_input; % Gaussian filtering over cortical space
V1_noise_input = gather(V1_noise_input); gpuDevice(1); toc
end