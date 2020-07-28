function [] = analysis_OP_correlation(CORR_PATTERN,OP_MAP,Scalevar,OP_loc,Pwl_loc)
tic; disp("OP difference - correlation strength plot");
figure;
subplot(131);
% Point set 1: OP domain
x1s = round(OP_loc(1,:)/2); y1s = round(OP_loc(2,:)/2);
Y1 = zeros(size(OP_loc,2),10);
for ii = 1:size(OP_loc,2)
    yy1 = y1s(ii); xx1 = x1s(ii);
    OPdiff1 = reshape(OP_MAP-OP_MAP(yy1,xx1),1,size(OP_MAP,1)*size(OP_MAP,2));
    OPdiff1 = abs(OPdiff1);
    OPdiff1(OPdiff1 > pi/2) = pi-OPdiff1(OPdiff1 > pi/2);
    OPdiff1 = abs(OPdiff1);
    OPdiff1 = OPdiff1*180/pi;
    OPdiff1 = round(OPdiff1/Scalevar)*Scalevar;
    
    CORRarray1 = reshape(CORR_PATTERN(yy1,xx1,:,:),1,size(OP_MAP,1)*size(OP_MAP,2));
    
    x1 = sort(unique(OPdiff1)); y1 = zeros(size(x1)); yn = zeros(size(x1));
    for jj = 1:length(CORRarray1)
        [~,ind] = find(OPdiff1(jj) == x1);
        yn(ind) = yn(ind)+1;
        y1(ind) = y1(ind)+CORRarray1(jj);
    end
    y1 = y1./yn;
    Y1(ii,:) = y1;
    plot(x1,y1,'-x','Color',[0.8 0.8 0.8]); hold on;
end

% Correlation test
[R1,P1,RL,RU] = corrcoef(x1,mean(Y1,1),'Alpha',0.05);

errorbar(x1,mean(Y1,1),std(Y1,1),'-bx','linewidth',2,'markersize',8);
xlabel("OP difference from seed point ("+char(176)+")"); xlim([-5 95]); xticks([0 30 60 90]);
ylabel("Spontaneous activity correlation"); ylim([-0.2 0.2]); yticks([-0.2 0 0.2]);
title("OP domains");

subplot(132);
% Point set 2: Pinwheels
x2s = round(Pwl_loc(1,:)/2); y2s = round(Pwl_loc(2,:)/2);
Y2 = zeros(size(Pwl_loc,2),10);
for ii = 1:size(Pwl_loc,2)
    yy2 = y2s(ii); xx2 = x2s(ii);
    OPdiff2 = reshape(OP_MAP-OP_MAP(yy2,xx2),1,size(OP_MAP,1)*size(OP_MAP,2));
    OPdiff2 = abs(OPdiff2);
    OPdiff2(OPdiff2 > pi/2) = pi-OPdiff2(OPdiff2 > pi/2);
    OPdiff2 = abs(OPdiff2);
    OPdiff2 = OPdiff2*180/pi;
    OPdiff2 = round(OPdiff2/Scalevar)*Scalevar;
    
    CORRarray2 = reshape(CORR_PATTERN(yy2,xx2,:,:),1,size(OP_MAP,1)*size(OP_MAP,2));
    
    x2 = sort(unique(OPdiff2)); y2 = zeros(size(x2)); yn = zeros(size(x2));
    for jj = 1:length(CORRarray2)
        [~,ind] = find(OPdiff2(jj) == x2);
        yn(ind) = yn(ind)+1;
        y2(ind) = y2(ind)+CORRarray2(jj);
    end
    y2 = y2./yn;
    Y2(ii,:) = y2;
    plot(x2,y2,'-o','Color',[0.8 0.8 0.8]); hold on;
end

% Correlation test
[R2,P2,RL,RU] = corrcoef(x2,mean(Y2,1),'Alpha',0.05);

errorbar(x2,mean(Y2,1),std(Y2,1),'-ro','linewidth',2,'markersize',8);
xlabel("OP difference from seed point ("+char(176)+")"); xlim([-5 95]); xticks([0 30 60 90]);
ylabel("Spontaneous activity correlation"); ylim([-0.2 0.2]); yticks([-0.2 0 0.2]);
title("Pinwheels");

subplot(133);
errorbar(x1-0.5,mean(Y1,1),std(Y1,1),'-bx','linewidth',2,'markersize',8); hold on;
errorbar(x2+0.5,mean(Y2,1),std(Y2,1),'-ro','linewidth',2,'markersize',8);
xlabel("OP difference from seed point ("+char(176)+")"); xlim([-5 95]); xticks([0 30 60 90]);
ylabel("Spontaneous activity correlation"); ylim([-0.2 0.2]); yticks([-0.2 0 0.2]);
title("Model, Pearson correlation test"); legend(sprintf("OP domains, p = %.6f",P1(1,2)),sprintf("Pinwheels, p = %.3f",P2(1,2)));

% Significance testing: Wilcoxon signed rank test
Rank1 = zeros(1,10);
Rank2 = zeros(1,10);
for ii = 1:10
    Rank1(ii) = signrank(Y1(:,ii));
    Rank2(ii) = signrank(Y2(:,ii));
end

Rank1
Rank2
toc
end