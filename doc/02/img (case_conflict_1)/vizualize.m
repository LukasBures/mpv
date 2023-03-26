%%
close all;
clear all;
clc;

%%
X = [1, 2, 3, 4, 2, 1, 3, 5, 1, 3, 2, 1, 5, 2, 2, 3, 4, 4, 1, 5, 8, 7, 5, 6, 4, 8, 1];
Y = [1, 1, 2, 2, 3, 4, 4, 3, 1, 3, 2, 1, 4, 1, 2, 4, 3, 4, 5, 5, 6, 4, 7, 8, 9, 1, 9];

cX = 8;
cY = 6;

%%
h = figure();
ang = 0:0.01:2 * pi; 
r = 3;
xp = r * cos(ang);
yp = r * sin(ang);
plot(cX + xp, cY + yp, 'MarkerSize', 10, 'LineWidth', 2);
axis equal;
hold on;
plot(X, Y, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlim([-1, 11.5]);
ylim([-1, 11.5]);


%%

mPos = 1;
navstivene = 0;
idxNavstivene = 1;
stredy = zeros(100, 2);
while true
    
    sumX = 0;
    sumY = 0;
    n = 0;
    for i = 1:1:length(X)
        aX = X(i);
        aY = Y(i);
        if (sqrt((aX - cX)^2 + (aY - cY)^2)) < r
            sumX = sumX + aX;        
            sumY = sumY + aY;        
            n = n + 1;
            navstivene(idxNavstivene) = i;
            idxNavstivene = idxNavstivene + 1;
        end
    end

    mX(mPos) = sumX / n;
    mY(mPos) = sumY / n;
    
    stredy(mPos, :) = [mX(mPos), mY(mPos)];
    
    
    if(mPos > 1) && (mX(mPos) == mX(mPos - 1)) && (mY(mPos) == mY(mPos - 1))
        break;
    end
    cX = mX(mPos);
    cY = mY(mPos);
    mPos = mPos + 1;
end

plot(stredy(1:mPos-1, 1), stredy(1:mPos-1, 2), 'm+', 'MarkerSize', 10, 'LineWidth', 2);
plot(stredy(1:mPos-1, 1), stredy(1:mPos-1, 2), 'k--', 'MarkerSize', 10, 'LineWidth', 2);
plot(X(navstivene), Y(navstivene), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(stredy(mPos-1, 1) + xp, stredy(mPos-1, 2) + yp, 'b--', 'MarkerSize', 10, 'LineWidth', 2);
legend('Startovaci okenko', '2D body', 'Stredni hodnoty', 'Trajektorie konvergence', ...
    'Navstivene body', 'Okenko po konvergenci', 4);
xlabel('x_1', 'FontSize', 15);
ylabel('x_2', 'FontSize', 15);
title('Mean Shift - tvorba shluku', 'FontSize', 15, 'fontWeight', 'bold');


%%
print(h, '-dpdf', 'MeanShift');









