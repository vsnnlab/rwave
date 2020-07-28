data = {'2020-03-23,15-26', '2020-03-23,15-29', '2020-03-23,15-32',...
    '2020-03-23,15-36', '2020-03-23,15-39', '2020-03-23,15-42', '2020-03-23,15-45',...
    '2020-03-23,15-48', '2020-03-23,15-51', '2020-03-23,15-55', '2020-03-23,15-58',...
    '2020-03-23,16-01', '2020-03-23,16-04', '2020-03-23,16-08', '2020-03-23,16-11',...
    '2020-03-23,16-14', '2020-03-23,16-17', '2020-03-23,16-21', '2020-03-23,16-24',...
    '2020-03-23,16-27'};
data = {'2020-03-24,15-55', '2020-03-24,16-06', '2020-03-24,16-16',...
    '2020-03-24,16-26', '2020-03-24,16-37', '2020-03-24,16-47', '2020-03-24,16-57',...
    '2020-03-24,17-08', '2020-03-24,17-18', '2020-03-24,17-28', '2020-03-24,17-38',...
    '2020-03-24,17-48', '2020-03-24,17-58', '2020-03-24,18-09', '2020-03-24,18-20',...
    '2020-03-24,18-31', '2020-03-24,18-41', '2020-03-24,18-52', '2020-03-24,19-03',...
    '2020-03-24,19-13'};
w = zeros(0, 0);
initial_idx = zeros(0);

for i = 1:length(data)
    weight_path = strcat('./export/Monkey/',cell2mat(data(i)),'/data_weight_matrix/w_epoch');
    initial_idx(i) = 31*(i-1)+1;
    load(strcat(weight_path, "0"));
    w(2*(i-1)+1, :) = w_V1_V1(:);
    load(strcat(weight_path, "8"));
    w(2*i, :) = w_V1_V1(:);
end

C = nchoosek(1:length(data),2);

N = size(C,1);
init_corr = zeros(N, 1);
final_corr = zeros(N, 1);

for i = 1:N
    pair = C(i,:);
    init1 = pair(1);
    init2 = pair(2);
    
    w1_init = w(2*(init1-1)+1,:);
    w1_final = w(2*init1,:);
    w2_init = w(2*(init2-1)+1,:);
    w2_final = w(2*init2,:);
    
    w1_init = remove_diagonal(w1_init);
    w1_final = remove_diagonal(w1_final);
    w2_init = remove_diagonal(w2_init);
    w2_final = remove_diagonal(w2_final);
    
    [r_init, p] = corrcoef(w1_init, w2_init);
    [r_final, p] = corrcoef(w1_final, w2_final);
    
    init_corr(i) = r_init(2,1);
    final_corr(i) = r_final(2,1);
end

[h, p1] = ttest(init_corr);
[h, p2] = ttest(final_corr);
[h, p3] = ttest(init_corr, final_corr);
[h, p4] = ttest(final_corr-1);

figure;
errorbar([1,2],[mean(init_corr),mean(final_corr)], [std(init_corr),std(final_corr)]);
xlim([0.5 2.5]); ylim([-0.2 1.2]);
suptitle("initial p = " + num2str(p1) + ...
    ", final p = " + num2str(p2) + ...
    ", initial-final p = " + num2str(p3) + ...
    ", final p from 1 = " + num2str(p4));

disp(num2str(mean(init_corr)) + "+-" + num2str(std(init_corr)) + ...
    " -> " + num2str(mean(final_corr)) + "+-" + num2str(std(final_corr)))

function d = remove_diagonal(t)
    t = reshape(t, [sqrt(size(t,2)), sqrt(size(t,2))]);
    d = reshape(t(~diag(ones(1,size(t,1)))), size(t)-[1 0]);
    size(d)
end


% N = 50;
% for i = 1:N
%     pair = randsample(5, 2);
%     init1 = pair(1);
%     init2 = pair(2);
%     
%     w1_init = w(2*(init1-1)+1,:);
%     w1_final = w(2*init1,:);
%     w2_init = w(2*(init2-1)+1,:);
%     w2_final = w(2*init2,:);
%     
%     r_init = corrcoef(w1_init, w2_init);
%     r_final = corrcoef(w1_final, w2_final);
%     
%     init_corr(i) = r_init(2,1);
%     final_corr(i) = r_final(2,1);
% end
% 
% [h, p] = ttest(init_corr, final_corr);
% 
% figure;
% errorbar([1,2],[mean(init_corr),mean(final_corr)], [std(init_corr),std(final_corr)]);
% xlim([0.5 2.5]); ylim([-0.2 1.2]);
% suptitle("paired t-test, p = " + num2str(p));