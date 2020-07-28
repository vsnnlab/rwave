% Paths
addpath('./function_horizontal_learning/');
addpath('./function_init_model/');
addpath('./function_map_analysis/');
addpath('./function_utils/');
addpath('./export/');

folderDir = './export/2020-03-13,20-28/'
weight_path = strcat(folderDir,'/data_weight_matrix/w_epoch');

Result = compute_OP(pos_ON,pos_OFF,w0_V1_ON,w0_V1_OFF,w_V1_ON,w_V1_OFF);
op0 = Result(:,1);
op = Result(:,2);

n_V1 = size(op, 1);

rng(96); % 88, 90 or 96
V1_ind = datasample(1:n_V1,20,'Replace',false);
op_img = op(V1_ind);
[~, V1_ind_sorted] = sort(op_img);

figure;

load(strcat(weight_path, '0'));
w_img = w_V1_V1(V1_ind_sorted, V1_ind_sorted);
w_img(w_img==0) = nan;
subplot(231); imagesc(w_img); axis image xy; colorbar;
title('epoch 0');

load(strcat(weight_path, '15'));
w_img = w_V1_V1(V1_ind_sorted, V1_ind_sorted);
w_img(w_img==0) = nan;
subplot(232); imagesc(w_img); axis image xy; colorbar;
title('epoch 15');

load(strcat(weight_path, '30'));
w_img = w_V1_V1(V1_ind_sorted, V1_ind_sorted);
w_img(w_img==0) = nan;
subplot(233); imagesc(w_img); axis image xy; colorbar;
title('epoch 30');

load(strcat(weight_path, '0'));
w_img = wp_V1_V1(V1_ind_sorted, V1_ind_sorted);
w_img(w_img==0) = nan;
subplot(234); imagesc(w_img); axis image xy; colorbar;
title('epoch 0');

load(strcat(weight_path, '15'));
w_img = wp_V1_V1(V1_ind_sorted, V1_ind_sorted);
w_img(w_img==0) = nan;
subplot(235); imagesc(w_img); axis image xy; colorbar;
title('epoch 15');

load(strcat(weight_path, '30'));
w_img = wp_V1_V1(V1_ind_sorted, V1_ind_sorted);
w_img(w_img==0) = nan;
subplot(236); imagesc(w_img); axis image xy; colorbar;
title('epoch 30');