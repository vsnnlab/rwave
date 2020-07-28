function [Result] = init_horizontal(pos_V1,rnn_r_lim)
rng('shuffle');
parallel.gpu.rng('shuffle', 'Philox4x32-10');
tic; n_V1 = size(pos_V1,1);
% Random initialization
w0_V1_V1 = gather(gpuArray.randn(n_V1,n_V1))*0.1+1;
w0_V1_V1(w0_V1_V1<0) = 0;
% No neuron wired to itself
w0_V1_V1 = w0_V1_V1-diag(diag(w0_V1_V1));

distx = pos_V1(:,1)-pos_V1(:,1)';
disty = pos_V1(:,2)-pos_V1(:,2)';

dist2 = distx.^2+disty.^2<rnn_r_lim^2;

% Output normalization
w0_V1_V1 = w0_V1_V1.*(sum(w0_V1_V1,1).^-1);
Result = {gather(w0_V1_V1)};
gpuDevice(1); toc
end