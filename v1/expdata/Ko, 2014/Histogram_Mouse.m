figure;
subplot(141); hold on;
bar(1,58/353); bar(2,64/295,'b'); bar(3,44/233,'r');
xticks([1 2 3]); xticklabels({'P13-P15','P22-P26','Dark reared'});
yticks([0 0.1 0.2 0.3 0.4 0.5]); xlim([0 4]); ylim([0 0.5]); title("p = 0.23");
subplot(142); hold on;
bar([1 2 3],[8/34 5/35 3/16]);
xticks([1 2 3]); xticklabels({'0','45','90'});
yticks([0 0.1 0.2 0.3 0.4 0.5]); xlim([0 4]); ylim([0 0.5]); title("P13-15, P = 0.27");
subplot(143); hold on;
bar([1 2 3],[9/25 15/60 4/28],'b');
xticks([1 2 3]); xticklabels({'0','45','90'});
yticks([0 0.1 0.2 0.3 0.4 0.5]); xlim([0 4]); ylim([0 0.5]); title("P22-26, p = 0.034");
subplot(144); hold on;
bar([1 2 3],[8/26 6/49 4/35],'r');
xticks([1 2 3]); xticklabels({'0','45','90'});
yticks([0 0.1 0.2 0.3 0.4 0.5]); xlim([0 4]); ylim([0 0.5]); title("Dark reared, p = 0.028");