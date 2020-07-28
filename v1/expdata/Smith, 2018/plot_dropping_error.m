ERR = load("Data/MeanError.csv");
U = load("Data/upper.csv");
L = load("Data/lower.csv");

figure;
errorbar(ERR(1:end-1,1),ERR(1:end-1,2),(U(1:end-1,2)-L(1:end-1,2))/2,"-ok","MarkerSize",8,"LineWidth",2,"MarkerFaceColor","k");
hold on;
plot([-9 6],[45 45],":k","LineWidth",2);
pbaspect([1 1 1]);
area([-1 1],[50 50],"FaceColor","g","EdgeColor","g",'FaceAlpha',.3,'EdgeAlpha',.3);

xlabel("Age relative to EO (days)"); xlim([-9 1]);
xticks([-10 -5 0]);
ylabel("Orientation prediction error ("+char(176)+")");
ylim([0 46]); yticks([0 15 30 45]);
title("Group mean");
