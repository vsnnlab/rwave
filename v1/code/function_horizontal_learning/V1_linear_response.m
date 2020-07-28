function [V1_input] = V1_linear_response(wave_ON,wave_OFF,weight_V1_ON,weight_V1_OFF)
% Load small arrays on GPU
ON_output = gpuArray(wave_ON)'; OFF_output = gpuArray(wave_OFF)';
% V1 feedforward inputs
weight_V1_ON = gpuArray(weight_V1_ON); weight_V1_OFF = gpuArray(weight_V1_OFF);
% Retrieve
V1_input = gather((weight_V1_ON*ON_output + weight_V1_OFF*OFF_output)');
end