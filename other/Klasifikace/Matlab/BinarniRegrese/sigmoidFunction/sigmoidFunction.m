

% Data 
x = -6:0.01:6;


y = 1 ./ (1 + exp(- x));


% Vykresleni
h = figure;
plot(x, y, 'r-', 'MarkerSize', 10, 'LineWidth', 2);
grid on;
xlabel('z');
legend('Sigmoid funkce g(z)', 'FontSize', 12, 4);
% ulozeni do souboru
print(h, '-dpdf', 'sigmoidFunction');