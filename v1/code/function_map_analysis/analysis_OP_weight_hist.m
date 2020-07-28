function[p] = analysis_OP_weight_hist(OP,w_V1_V1,pos_V1)

OPdiff = abs(OP'-OP);
OPdiff(OPdiff>pi/2) = pi-OPdiff(OPdiff>pi/2);

OP_categories = [pi*(0+1)/24 pi*(1+2)/24 pi*(2+3)/24 pi*(3+4)/24 pi*(4+5)/24 pi*(5+6)/24];
OPdiff(OPdiff>=pi*0/12 & OPdiff<pi*1/12) = OP_categories(1);
OPdiff(OPdiff>=pi*1/12 & OPdiff<pi*2/12) = OP_categories(2);
OPdiff(OPdiff>=pi*2/12 & OPdiff<pi*3/12) = OP_categories(3);
OPdiff(OPdiff>=pi*3/12 & OPdiff<pi*4/12) = OP_categories(4);
OPdiff(OPdiff>=pi*4/12 & OPdiff<pi*5/12) = OP_categories(5);
OPdiff(OPdiff>=pi*5/12 & OPdiff<pi*6/12) = OP_categories(6);

dist2 = (pos_V1(:,1)-pos_V1(:,1)').^2 + (pos_V1(:,2)-pos_V1(:,2)').^2;
mask = dist2 >= 200^2;
w_V1_V1 = w_V1_V1.*mask;

mean_weight = zeros(1,length(OP_categories));
std_weight = zeros(1,length(OP_categories));
weight_anova = []; label_anova = [];
for ii = 1:length(OP_categories)
    weight = w_V1_V1.*(OPdiff == OP_categories(ii));
    weight(weight==0) = [];
    weight_anova = [weight_anova weight];
    label_anova = [label_anova repmat(ii,1,length(weight))];
    mean_weight(ii) = mean(weight);
    std_weight(ii) = std(weight);
end
FINAL = gather(mean_weight)/max(mean_weight); FINAL_STD = gather(std_weight)/max(mean_weight);

p = cuzick([weight_anova' label_anova']);

hold on;
bar(OP_categories/pi*180,FINAL,'r');
errorbar(OP_categories/pi*180,FINAL,FINAL_STD,'.k');
yticks([0 0.5 1]); xlabel("OP difference (degree)"); ylabel("Connection weight");
legend("p="+num2str(p),"location","southwest");
title("# of connections = "+num2str(size(weight_anova,2)));
end