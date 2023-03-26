
ccc;

%%
dataX = [0.4, 0.6, 0.9 1.0, 1.5, 2.2, 3.0 3.9 5];
dataY = [0.4, 1.2, 3.8 2.2, 3.0, 3.4, 3.9 4.1 5.2];

krok = 0:0.01:ceil(dataX(end));

%% VYKRESLENI DAT
h = figure;
plot(dataX, dataY, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
xlabel('Podlahova plocha v m^2', 'FontSize', 12);
ylabel('Cena bytu v tis. Kc', 'FontSize', 12);
xlim([0, ceil(dataX(end))]);
ylim([0, ceil(dataY(end))]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
legend('Data', 4);


%
print(h, '-dpdf', 'regrese_0');

%% LINEARNI
p = polyfit(dataX, dataY, 1);
f = polyval(p, krok);

h = figure;
plot(dataX, dataY, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(krok, f, 'b-', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Podlahova plocha v m^2', 'FontSize', 12);
ylabel('Cena bytu v tis. Kc', 'FontSize', 12);
xlim([0, ceil(dataX(end))]);
ylim([0, ceil(dataY(end))]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
legend('Data', 'Hypoteza: h_{\Theta}(x) = \vartheta_0 + \vartheta_1 x', 4);

%
print(h, '-dpdf', 'regrese_1');

%% KVADRATICKA
p = polyfit(dataX, dataY, 2);
f = polyval(p, krok);

h = figure;
plot(dataX, dataY, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(krok, f, 'b-', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Podlahova plocha v m^2', 'FontSize', 12);
ylabel('Cena bytu v tis. Kc', 'FontSize', 12);
xlim([0, ceil(dataX(end))]);
ylim([0, ceil(dataY(end))]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
legend('Data', 'Hypoteza: h_{\Theta}(x) = \vartheta_0 + \vartheta_1 x + \vartheta_2 x^2', 4);

%
print(h, '-dpdf', 'regrese_2');

%% N-POLYNOM
p = polyfit(dataX, dataY, 4);
f = polyval(p, krok);

h = figure;
plot(dataX, dataY, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(krok, f, 'b-', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Podlahova plocha v m^2', 'FontSize', 12);
ylabel('Cena bytu v tis. Kc', 'FontSize', 12);
xlim([0, ceil(dataX(end))]);
ylim([0, ceil(dataY(end))]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
legend('Data', 'Hypoteza: h_{\Theta}(x) = \vartheta_0 + \vartheta_1 x + \vartheta_2 x^2 + \vartheta_3 x^3 + \vartheta_4 x^4', 4);

%
print(h, '-dpdf', 'regrese_4');

%% UNDERFIT

h = figure;
plot(dataX, dataY, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot([krok(1), krok(end)], [mean(dataY), mean(dataY)], 'b-', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Podlahova plocha v m^2', 'FontSize', 12);
ylabel('Cena bytu v tis. Kc', 'FontSize', 12);
xlim([0, ceil(dataX(end))]);
ylim([0, ceil(dataY(end))]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
legend('Data', 'Hypoteza: h_{\Theta}(x) = \vartheta_0', 4);

%
print(h, '-dpdf', 'regrese_underfit');


