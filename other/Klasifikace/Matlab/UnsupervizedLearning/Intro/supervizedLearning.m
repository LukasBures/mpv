

close all;
clear all;
clc;
 
%% DATA DEFINITION
dataX1 = [0.7 1.1 1 1.5 1.1];
dataY1 = [1 0.5 1.6 0.9 0.95];
dataX2 = 2 + [0.7 1.2 1 1.5 1.1];
dataY2 = 2 + [1 0.8 1.4 0.9 0.95];

lineX = [0 4];
lineY = [4 0];

circleX = [1 3];
circleY = [1 3];


%% PLOT supervized 

h = figure;
plot(dataX1, dataY1, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(dataX2, dataY2, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
xlim([0 4]);
ylim([0 4]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
print(h, '-dpdf', 'supervized');

%% PLOT supervized + line

h = figure;
plot(dataX1, dataY1, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(dataX2, dataY2, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(lineX, lineY, 'k-', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
xlim([0 4]);
ylim([0 4]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
print(h, '-dpdf', 'supervized_line');

%% PLOT unsupervized

h = figure;
plot(dataX1, dataY1, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(dataX2, dataY2, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
xlim([0 4]);
ylim([0 4]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
print(h, '-dpdf', 'unsupervized');

%% PLOT unsupervized + circle

h = figure;
plot([dataX1 , dataX2], [dataY1, dataY2], 'ko', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(-1, -1, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
plot(-1, -1, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% plot(dataX2, dataY2, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
plot(circleX(1), circleY(1), 'bo', 'MarkerSize', 140, 'LineWidth', 2);
plot(circleX(2), circleY(2), 'ro', 'MarkerSize', 140, 'LineWidth', 2);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
xlim([0 4]);
ylim([0 4]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
legend('Data', 'Shluk 1', 'Shluk 2', 4);
print(h, '-dpdf', 'unsupervized_circle');







