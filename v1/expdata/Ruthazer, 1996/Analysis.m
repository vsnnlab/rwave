CI = load("CI.csv");
ST = load("ST.csv");
OT = load("OT.csv");
L = load("Lower_std.csv"); L = L(:,2);
U = load("Upper_std.csv"); U = U(:,2);
STD = (U-L)/2;

LIM = 40;

gcf = figure; set(gcf,'defaultAxesColorOrder',[[0 0 0];[0 0 0]]);
yyaxis left;
Range = CI(:,1)<40;
errorbar(CI(Range,1),CI(Range,2),STD(Range),"-sk","Linewidth",1.5,"Markersize",10,'MarkerFaceColor','k');
xlim([19 LIM+1]); xticks([20 25 30 35 40]); ylim([-0.3 1.8]); yticks([0 0.5 1.0 1.5]);
xlabel("Age (postnatal days)"); ylabel("Cluster index");

yyaxis right;
Range = ST(:,1)<40;
plot(ST(Range,1),ST(Range,2),":or","Linewidth",1.5,"Markersize",7);
hold on;
Range = OT(:,1)<40;
plot(OT(Range,1),OT(Range,2),"--^b","Linewidth",1.5,"Markersize",7);
ylim([9 50]); yticks([15 25 35 45]); ylabel("Single-unit OSI")

pbaspect([1 1 1]);
area([28 32],[50 50],"FaceColor","g","EdgeColor","g",'FaceAlpha',.3,'EdgeAlpha',.3);
legend("Cluster index","Single-unit tuning","Optical tuning","Eye opening","location","northwest");

figure;
errorbar([1 2],[CI(1,2) CI(3,2)],[STD(1) STD(3)]);
xlim([0.5 2.5]);
title("1996 Ruthazer data, P21-P27 CI");