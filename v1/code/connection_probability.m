% require: op, w0_V1_V1, w_V1_V1

OPdiff = abs(op'-op);
OPdiff(OPdiff>pi/2) = pi-OPdiff(OPdiff>pi/2);

OP_categories = [pi*(0+1)/12 pi*(1+2)/12 pi*(2+3)/12];
OPdiff(OPdiff>=pi*0/6 & OPdiff<pi*1/6) = OP_categories(1);
OPdiff(OPdiff>=pi*1/6 & OPdiff<pi*2/6) = OP_categories(2);
OPdiff(OPdiff>=pi*2/6 & OPdiff<pi*3/6) = OP_categories(3);

w_lim = max(w_V1_V1(:));

w_V1_V1_n = w_V1_V1/w_lim; % normalize [0,1] -> use as connection probability
threshold = 1-1e-4;
coupling = threshold <= w_V1_V1_n;

pair_cnt = zeros(1,length(OP_categories));
connection_cnt = zeros(1,length(OP_categories));
for ii = 1:length(OP_categories)
    pair_cnt(ii) = sum(OPdiff(:) == OP_categories(ii));
    connection_cnt(ii) = sum(coupling(OPdiff == OP_categories(ii)));
end
pair_cnt(1) = pair_cnt(1) - size(w_V1_V1, 1); % exclude recurrent connections

[p_CA,stats]=cochran_arm([pair_cnt;connection_cnt],OP_categories);

p = connection_cnt./pair_cnt;
figure; plot(OP_categories,p); title("threshold = "+num2str(threshold)+"*w_{lim}, pval="+num2str(p_CA));
xticks(OP_categories);
