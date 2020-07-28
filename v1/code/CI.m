% Paths
addpath('./function_horizontal_learning/');
addpath('./function_init_model/');
addpath('./function_map_analysis/');
addpath('./function_utils/');
addpath('./export/');

weight_path = './export/2020-02-02,01-28/data_weight_matrix/w_epoch';

load(strcat(weight_path, '15'));
w_15_V1_V1 = w_V1_V1;
wp_15_V1_V1 = wp_V1_V1;

load(strcat(weight_path, '30'));
w_30_V1_V1 = w_V1_V1;
wp_30_V1_V1 = wp_V1_V1;

% Spatial clustering analysis: Hopkins' ratio statistics
CI_initial = analysis_clustering_Hopkins(pos_V1,w0_V1_V1,d_OFF,crop_x,crop_y);
CI_15_clustered = analysis_clustering_Hopkins(pos_V1,w_15_V1_V1,d_OFF,crop_x,crop_y);
CI_15_permuted = analysis_clustering_Hopkins(pos_V1,wp_15_V1_V1,d_OFF,crop_x,crop_y);
CI_30_clustered = analysis_clustering_Hopkins(pos_V1,w_30_V1_V1,d_OFF,crop_x,crop_y);
CI_30_permuted = analysis_clustering_Hopkins(pos_V1,wp_30_V1_V1,d_OFF,crop_x,crop_y);

figure;
subplot(141);
[h,p] = ttest2(CI_initial',CI_15_clustered');
hold on; bar([1 2],[mean(CI_initial),mean(CI_15_clustered)]);
errorbar([1 2],[mean(CI_initial),mean(CI_15_clustered)],[std(CI_initial),std(CI_15_clustered)],'.k');
title(sprintf("Clustering index (CI), N = "+num2str(size(CI_initial,2))+", t-test p = "+num2str(p))); drawnow;
ylim([-0.5 1.5]);
subplot(143);
[h,p] = ttest2(CI_initial',CI_15_permuted');
hold on; bar([1 2],[mean(CI_initial),mean(CI_15_permuted)]);
errorbar([1 2],[mean(CI_initial),mean(CI_15_permuted)],[std(CI_initial),std(CI_15_permuted)],'.k');
title(sprintf("Permuted wave, N = "+num2str(size(CI_initial,2))+ ", t-test p = "+num2str(p))); drawnow;
ylim([-0.5 1.5]);

subplot(142);
[h,p] = ttest2(CI_15_clustered',CI_30_clustered');
hold on; bar([1 2],[mean(CI_15_clustered),mean(CI_30_clustered)]);
errorbar([1 2],[mean(CI_15_clustered),mean(CI_30_clustered)],[std(CI_15_clustered),std(CI_30_clustered)],'.k');
title(sprintf("Clustering index (CI), N = "+num2str(size(CI_15_clustered,2))+", t-test p = "+num2str(p))); drawnow;
ylim([-0.5 1.5]);
subplot(144);
[h,p] = ttest2(CI_15_permuted',CI_30_permuted');
hold on; bar([1 2],[mean(CI_15_permuted),mean(CI_30_permuted)]);
errorbar([1 2],[mean(CI_15_permuted),mean(CI_30_permuted)],[std(CI_15_permuted),std(CI_30_permuted)],'.k');
title(sprintf("Permuted wave, N = "+num2str(size(CI_15_permuted,2))+ ", t-test p = "+num2str(p))); drawnow;
ylim([-0.5 1.5]);

%%
figure; [h,p] = ttest2(CI_15_permuted',CI_15_clustered');
subplot(121); hold on;
errorbar([1 2],[mean(CI_initial),mean(CI_15_clustered)],[std(CI_initial),std(CI_15_clustered)],'k');
errorbar([1 2],[mean(CI_initial),mean(CI_15_permuted)],[std(CI_initial),std(CI_15_permuted)],'r');
ylim([-1 2]);
title(sprintf("Clustering index (CI), N = "+num2str(size(CI_initial,2))+", t-test p = "+num2str(p))); drawnow;

[h,p] = ttest2(CI_30_permuted',CI_30_clustered');
subplot(122); hold on;
errorbar([1 2],[mean(CI_15_clustered),mean(CI_30_clustered)],[std(CI_15_clustered),std(CI_30_clustered)],'k');
errorbar([1 2],[mean(CI_15_permuted),mean(CI_30_permuted)],[std(CI_15_permuted),std(CI_30_permuted)],'r');
ylim([-1 2]);
title(sprintf("Clustering index (CI), N = "+num2str(size(CI_initial,2))+", t-test p = "+num2str(p))); drawnow;