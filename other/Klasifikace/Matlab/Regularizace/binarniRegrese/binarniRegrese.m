
ccc;

%% VYKRESLENI DAT
dataY1 = [1.0 1.0 1.0, 2.0 2.1, 2.0 2.0 1.5, 2.5 2.3, 4.0 2.7 3.2 5.0, 3.0];
dataX1 = [2.0 3.0 4.0, 3.8 4.7, 0.7 2.5 1.5, 1.2 1.9, 1.0 1.8 1.9 1.4, 3.0];
dataY2 = [2.5 2.5, 3.0 3.3 3.5 4.1 5.0, 3.9 4.3 4.6, 4.5 3.2 3.9 2.9 2.8 3.1 3.9 4.1];
dataX2 = [2.5 3.5, 2.6 2.4 3.0 2.1 3.0, 2.6 2.8 2.7, 4.8 3.8 2.9 3.6 4.1 3.7 4.8 4.1];

hraniceX = [1.50 2.0 2.3 2.4 2.5 3.0 3.5 3.8 4.1 4.8 5];
hraniceY = [10.0 4.1 3.3 2.4 2.4 3.4 2.4 2.3 2.7 2.8 3];

%
krok1 = 0:0.01:2.4;
p1 = polyfit(hraniceX(1:4), hraniceY(1:4), 3);
yi1 = polyval(p1, krok1);

krok2 = 2.4:0.01:5;
p2 = polyfit(hraniceX(4:end), hraniceY(4:end), 7);
yi2 = polyval(p2, krok2);

h = figure;
plot(dataX1, dataY1, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(dataX2, dataY2, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
plot(krok1, yi1, 'm-', 'MarkerSize', 10, 'LineWidth', 2);
plot(krok2, yi2, 'm-', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
xlim([0, 5]);
ylim([0, 5]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
legend('Trida 1', 'Trida 2', 'Rozdelovaci nadrovina', 3);
%
print(h, '-dpdf', 'regularizace_binarni_regrese');
dataX = zeros(25, 1);








