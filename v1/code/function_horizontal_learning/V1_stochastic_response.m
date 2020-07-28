function [V1_profile] = V1_stochastic_response(n_events,t_steps_max,div_thr,...
    V1_max,V1_slope,V1_thr,w_V1_V1,V1_point_input,V1_noise_input)
tic; disp("Measuring spontaneous activity response");
n_V1 = size(w_V1_V1,1);
V1_point_input = gpuArray(V1_point_input);
w_V1_V1 = gpuArray(w_V1_V1);
V1_profile = gpuArray(zeros(n_V1,n_events));

for event = 1:n_events
    input_V1 = gpuArray(zeros(n_V1,t_steps_max));
    rate_V1 = gpuArray(zeros(n_V1,t_steps_max));
    
    % Integrate until response divergence
    input_V1(:,1) = V1_noise_input(:,event) + V1_point_input(:,event);
    rate_V1(:,1) = V1_max*logsig((input_V1(:,1)-V1_thr)/V1_slope);
    
    for tt = 2:t_steps_max
        % Current input: background noise + local stimulus + recurrent input
        input_V1(:,tt) = V1_noise_input(:,event) + V1_point_input(:,event) + w_V1_V1*rate_V1(:,tt-1);
        rate_V1(:,tt) = V1_max*logsig((input_V1(:,tt)-V1_thr)/V1_slope);
        
        % If the response first diverges at T, take response profile at T-1
        if mean(rate_V1(:,tt)) >= div_thr
            V1_profile(:,event) = rate_V1(:,tt-1);
            disp(tt-1); break;
        elseif tt == t_steps_max
            V1_profile(:,event) = rate_V1(:,t_steps_max);
        end
    end
end
V1_profile = gather(V1_profile);
gpuDevice(1);
toc
end