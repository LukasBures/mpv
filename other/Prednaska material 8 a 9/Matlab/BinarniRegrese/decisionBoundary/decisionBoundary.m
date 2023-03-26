ccc;

%%
% Data 
x1 = [1.5 1.9 2 2.1 2.2 2.25 2.3 2.6 2.8 3 2.9 3.2];
x2 = [0.1 0.21 0.315 0.5 0.8 1.1 1.5 2];
y1 = [3.1 3.2 3.3 2.5 2.8 2.4 2.6 2.3 2.1 1.8 1.5 1.1];
y2 = [2.2 1.5 0.5 1 1.5 0.5 0.4 0.1];

% Vykresleni
h = figure;
plot(x1, y1, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(x2, y2, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
% plot([0 3], [3 0], 'm-', 'MarkerSize', 10, 'LineWidth', 2);
% grid on;
xlabel('x_1');
ylabel('x_2');
% legend('Sigmoid funkce g(z)', 'FontSize', 12, 4);
% ulozeni do souboru
print(h, '-dpdf', 'decisionBoundary1');



%%
% Vykresleni
h = figure;
plot([0 3], [3 0], 'm-', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(x1, y1, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(x2, y2, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Rozhodovací hranice');
% grid on;
xlabel('x_1');
ylabel('x_2');
% legend('Sigmoid funkce g(z)', 'FontSize', 12, 4);
% ulozeni do souboru
print(h, '-dpdf', 'decisionBoundary2');


%% Nelinearni rozhodovaci hranice

x1 = [0.5 1 1.5 1.5 1 0.5 -0.5 -1 -1.5 -1.5 -1 -0.5 1.5 0 -1.5 0];
x2 = [0.1 0.21 0.315 0.5 0.8 -0.1 -0.5 -0.5 -0.1 -0.2 0.6 0.15 0.158 0.92];
y1 = [1.5 1 0.5 -0.5 -1 -1.5 -1.5 -1 -0.5 0.5 1 1.5 0 -1.5 0 1.5];
y2 = [ -0.1  0.90 -0.1 -0.21 -0.315 0 0.5 -0.5  -0.6 -0.15 0.158 0.5 0 -0.1];

% Vykresleni
h = figure;
plot(x1, y1, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(x2, y2, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
xlim([-2 2]);
ylim([-2 2]);
axis equal;
xlabel('x_1');
ylabel('x_2');
% legend('Sigmoid funkce g(z)', 'FontSize', 12, 4);
% ulozeni do souboru
print(h, '-dpdf', 'decisionBoundary3');

%% Nelinearni rozhodovaci hranice Reseni

uhel = 0:0.01:2*pi;
x3 = cos(uhel);
y3 = sin(uhel);


% Vykresleni
h = figure;
plot(x3, y3, 'm-', 'MarkerSize', 10, 'LineWidth', 2);
legend('Rozhodovací hranice');
hold on;
plot(x1, y1, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(x2, y2, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
xlim([-2 2]);
ylim([-2 2]);
axis equal;
xlabel('x_1');
ylabel('x_2');
% legend('Sigmoid funkce g(z)', 'FontSize', 12, 4);
% ulozeni do souboru
print(h, '-dpdf', 'decisionBoundary4');















