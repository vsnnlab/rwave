figure;
subplot(221);
dist1 = load("Dist1.csv");
[dist,ind] = sort(dist1(:,1)); number = abs(dist1(ind,2));
plot(dist,number,'k','linewidth',3); pbaspect([1 1 1])
xlim([0.5 5]); ylim([0 1800]);
xlabel("Distance from injection site (mm)");
xticks([0.5 1 2 3 4 5]); xticklabels(["0.5" "1.0" "2.0" "3.0" "4.0" "5.0"]);
yticks([0 300 600 900 1200 1500 1800]); ylabel("Number of boutons");
title("Bouton distribution along preferred axis");

subplot(222);
SCT = load("Scatter.csv");
scatter(SCT(:,1),SCT(:,2),'dk',"linewidth",2);
xlim([0 4]); ylim([0 4]); axis image;
hold on; plot([0 4],[0 4],'--k');
xlabel("Max distance preferred axis (mm)");
ylabel("Max distance orthogonal axis (mm)");
title("Long-ranged bouton distributions");

subplot(2,2,[3,4]);
polar = load("Polar1.csv");
[theta,ind] = sort(polar(:,2)); theta = theta*pi/180;
rho = polar(ind,1);
polarplot([theta; theta(1)],[rho; rho(1)],'k','linewidth',3);
rticks([]); rlim([0 max(rho)]);
thetaticklabels(string(0:30:330)+char(176));
hold on; polarplot(0,0,'+k','markersize',10,'linewidth',3);
polarplot([0 pi],[max(rho) max(rho)],'--k');
%legend("Median (13 cases)");
title("Orientation specificity of bouton distributions");

set(gcf, 'Position', [50 50 700 700])

figure;
sorted = theta;
sorted(theta>pi) = sorted(theta>pi)-pi;
sorted(sorted>pi/2) = pi-sorted(sorted>pi/2);
[sorted, ind] = sort(sorted);
plot(sorted,rho(ind),'o');
xticks([0, pi/6, pi/3 pi/2]); xlim([0 pi/2]);

hold on;
f = fit(sorted, rho(ind), 'exp1')
plot(f,sorted,rho(ind));