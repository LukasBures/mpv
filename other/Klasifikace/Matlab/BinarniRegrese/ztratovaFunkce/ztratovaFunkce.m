ccc;

%% -ztratovaFunkce1-
x1 = 0:0.01:1;
y1 = -log(x1);

% Vykresleni
h = figure;
plot(x1, y1, 'b-', 'MarkerSize', 10, 'LineWidth', 2);

xlabel('h_{\Theta} (x)');
title('y = 1', 'FontSize', 12);
legend('-log(h_{\Theta}(x))', 'FontSize', 12, 1);
print(h, '-dpdf', 'ztratovaFunkce1');

%% -ztratovaFunkce2-
x2 = 0:0.01:1;
y2 = -log(1 - x2);

% Vykresleni
h = figure;
plot(x2, y2, 'r-', 'MarkerSize', 10, 'LineWidth', 2);

xlabel('h_{\Theta} (x)');
title('y = 1', 'FontSize', 12);
legend('-log(1-h_{\Theta}(x))', 'FontSize', 12, 2);
print(h, '-dpdf', 'ztratovaFunkce2');

%% -ztratovaFunkce3-
% Vykresleni
h = figure;
plot(x1, y1, 'b-', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(x2, y2, 'r-', 'MarkerSize', 10, 'LineWidth', 2);

xlabel('h_{\Theta} (x)');
% title('y = 1', 'FontSize', 12);
legend('-log(h_{\Theta}(x))', '-log(1-h_{\Theta}(x))', 'FontSize', 12, 0);
print(h, '-dpdf', 'ztratovaFunkce3');

