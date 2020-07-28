% Cat/2020-03-23,16-57
% Cat/2020-03-23,17-35
% Cat/2020-03-23,17-42

% Monkey/2020-03-24,15-55
% Monkey/2020-03-24,16-06
% Monkey/2020-03-24,16-16

string = './export/Monkey/2020-03-24,16-16/data_weight_matrix/w_epoch';
string = './export/Cat/2020-03-23,17-42/data_weight_matrix/w_epoch';
%string = './export/2020-02-22,17-36/data_weight_matrix/w_epoch';
load(strcat(string,'0'));
w0_V1_V1 = w_V1_V1;
load(strcat(string,'30'));

dist2 = (pos_V1(:,1)-pos_V1(:,1)').^2 + (pos_V1(:,2)-pos_V1(:,2)').^2;
lhc_lim = period_V1*1;
lhc_mask = dist2*retina_V1_ratio^2 >= lhc_lim^2;
%lhc_mask = ones(size(lhc_mask));

N = 10;
OP_discrete = linspace(-pi/2, pi/2, N+1);

w0_agg = zeros(N, N);
nums = zeros(N, N);
for i=1:N
    % pre: an op category
    pre = op>=OP_discrete(i) & op<OP_discrete(i+1);
    for j=1:N
        % post: an op category
        post = op>=OP_discrete(j) & op<OP_discrete(j+1);
        % nonzero connections from pre to post
        mask_pre_post = reshape(pre, 1, length(pre)).*post.*(w0_V1_V1 ~= 0).*lhc_mask;
        w = w0_V1_V1.*mask_pre_post;
        w_mean = sum(w(:))/sum(mask_pre_post(:));
        w0_agg(j, i) = w_mean;
        nums(i, j) = sum(mask_pre_post(:));
    end
end
disp(sum(sum(nums==0,1),2))
disp(mean(nums(:)))

w_agg = zeros(N, N);
for i=1:N
    % pre: an op category
    pre = op>=OP_discrete(i) & op<OP_discrete(i+1);
    for j=1:N
        % post: an op category
        post = op>=OP_discrete(j) & op<OP_discrete(j+1);
        % nonzero connections from pre to post
        mask_pre_post = reshape(pre, 1, length(pre)).*post.*(w_V1_V1 ~= 0).*lhc_mask;
        w = w_V1_V1.*mask_pre_post;
        w_mean = sum(w(:))/sum(mask_pre_post(:));
        w_agg(j, i) = w_mean;
    end
end

wp_agg = zeros(N, N);
for i=1:N
    % pre: an op category
    pre = op>=OP_discrete(i) & op<OP_discrete(i+1);
    for j=1:N
        % post: an op category
        post = op>=OP_discrete(j) & op<OP_discrete(j+1);
        % nonzero connections from pre to post
        mask_pre_post = reshape(pre, 1, length(pre)).*post.*(wp_V1_V1 ~= 0).*lhc_mask;
        w = wp_V1_V1.*mask_pre_post;
        w_mean = sum(w(:))/sum(mask_pre_post(:));
        wp_agg(j, i) = w_mean;
    end
end

figure;
subplot(131); imagesc(w0_agg);
axis image xy; xlabel('pre'); ylabel('post');
caxis([min(w0_agg(:))+0.25*(max(w0_agg(:))-min(w0_agg(:)))...
    min(w0_agg(:))+0.75*(max(w0_agg(:))-min(w0_agg(:)))])
h=colorbar; colormap(hot);
t=get(h,'Limits');
set(h,'Ticks',linspace(t(1),t(2),5));
subplot(132); imagesc(w_agg);
axis image xy; xlabel('pre'); ylabel('post');
caxis([min(w_agg(:))+0.25*(max(w_agg(:))-min(w_agg(:)))...
    min(w_agg(:))+0.75*(max(w_agg(:))-min(w_agg(:)))])
h=colorbar; colormap(hot);
t=get(h,'Limits');
set(h,'Ticks',linspace(t(1),t(2),5));
subplot(133); imagesc(wp_agg);
axis image xy; xlabel('pre'); ylabel('post');
caxis([min(wp_agg(:))+0.25*(max(wp_agg(:))-min(wp_agg(:)))...
    min(wp_agg(:))+0.75*(max(wp_agg(:))-min(wp_agg(:)))])
h=colorbar; colormap(hot);
t=get(h,'Limits');
set(h,'Ticks',linspace(t(1),t(2),5));
suptitle("Only lhc (>500um, cortical distance)");

figure;
subplot(131); analysis_OP_weight_hist(op,w0_V1_V1,pos_V1,retina_V1_ratio,lhc_lim);
subplot(132); analysis_OP_weight_hist(op,w_V1_V1,pos_V1,retina_V1_ratio,lhc_lim);
subplot(133); analysis_OP_weight_hist(op,wp_V1_V1,pos_V1,retina_V1_ratio,lhc_lim);