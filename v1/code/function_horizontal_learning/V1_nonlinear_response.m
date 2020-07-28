function [Result] = V1_nonlinear_response(weight_V1_V1,V1_input,curve_max,curve_thr,curve_slope)
% Load arrays on GPU
weight_init = gpuArray(weight_V1_V1);
V1_input = gpuArray(V1_input);
V1_output = gpuArray(zeros(size(V1_input)));
% Iterate over V1 inputs and sum with lateral input to make whole V1 nonlinear response set
temporal_1D = size(V1_input,1);
for tt = 1:temporal_1D
    if tt~=1
        % Feedforward input + recurrent input + decaying past input
        V1_input(tt,:) = V1_input(tt,:) + (weight_init*V1_output(tt-1,:)')';% + V1_input(tt-1,:)*exp(-1/V1_TAU);
    end
    % Apply nonlinearity
    V1_output(tt,:) = curve_max*logsig((V1_input(tt,:)-curve_thr)/curve_slope);
end
% Retrieve
Result = gather(V1_output);
end