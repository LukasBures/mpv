%% Data 
x1 = [2 3 1.5 2.3 3 3.3 1.8 2.4];
x2 = [2 2.5 3 2.8 3 2.4 3.2 2.6];
x3 = [3 3.5 4 2.3];
x4 = [3 4.3 5];
y1 = [5 6 5.3 5.8 7.2 7.8 6 7];
y2 = [2 2.5 3 0.5 1 1.5 2 1];
y3 = [4 4.1 6.3 4.8];
y4 = [5.3 5.2 5.8];

%% Vykresleni - SVM img01
h = figure;
hold on;
line([1 6.5],[3.5 9],'Color', 'k','LineWidth', 1);
line([1 6.5],[2.5 8],'Color', 'k','LineWidth', 1);
line([1 6.5],[1.5 7],'Color', 'k','LineWidth', 1);
plot(x2+1, y2+1, 'x', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
plot(x1+1, y1+1, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');
%text
text(2.5,8,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(2.8,1.5,'$\omega_2$','interpreter','latex', 'FontSize', 20);
text(0.3,3.3,'$\displaystyle g_1\left( x \right)$','interpreter','latex', 'FontSize', 15);
text(0.3,2.3,'$\displaystyle g_2\left( x \right)$','interpreter','latex', 'FontSize', 15);
text(0.3,1.3,'$\displaystyle g_3\left( x \right)$','interpreter','latex', 'FontSize', 15);
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([-0.5, 7]);
ylim([0, 10]);
hold off

%Ulozeni
print(h, '-dpdf', 'SVM01');

%% Vykresleni - SVM img02
h = figure;
hold on;
line([1 6.5],[4 9.5],'Color', 'k','LineWidth', 1, 'LineStyle', '--');
line([1 6.5],[1 6.5],'Color', 'k','LineWidth', 1, 'LineStyle', '--');
line([1 6.5],[2.5 8],'Color', 'k','LineWidth', 2);
line([6 5.5],[7.5 8.5],'Color', 'k','LineWidth', 1);
line([6 5.5],[6 7],'Color', 'k','LineWidth', 1);
plot(x2+1, y2+1, 'x', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
plot(x1+1, y1+1, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');
plot([x1(1) x1(2)]+1, [y1(1) y1(2)]+1, 'ko', 'MarkerSize', 18);
plot([x2(1) x2(2) x2(3)]+1, [y2(1) y2(2) y2(3)]+1, 'ko', 'MarkerSize', 18);
%text
text(2.5,8,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(2.8,1.5,'$\omega_2$','interpreter','latex', 'FontSize', 20);
text(5.8,8.4,'$\displaystyle\frac{1}{\parallel\omega\parallel}$','interpreter','latex', 'FontSize', 13);
text(5.85,6.95,'$\displaystyle\frac{1}{\parallel\omega\parallel}$','interpreter','latex', 'FontSize', 13);
text(0,3.8,'$\displaystyle\omega^Tx+\omega_0=1$','interpreter','latex', 'FontSize', 15);
text(0,2.3,'$\displaystyle\omega^Tx+\omega_0=0$','interpreter','latex', 'FontSize', 15);
text(0,0.8,'$\displaystyle\omega^Tx+\omega_0=-1$','interpreter','latex', 'FontSize', 15);
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([-0.5, 7]);
ylim([0, 10]);
hold off

%Ulozeni
print(h, '-dpdf', 'SVM02');

%% Vykresleni - SVM img03
h = figure;
hold on;
line([1 6.5],[4 9.5],'Color', 'k','LineWidth', 1, 'LineStyle', '--');
line([1 6.5],[1 6.5],'Color', 'k','LineWidth', 1, 'LineStyle', '--');
line([1 6.5],[2.5 8],'Color', 'k','LineWidth', 1);
plot(x2+1, y2+1, 'x', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
plot(x1+1, y1+1, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');
plot(x3, y3, 'x', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
plot(x4, y4, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');
plot([x3(4) x3(3) x4(2) x4(3)], [y3(4) y3(3) y4(2) y4(3)], 'ko', 'MarkerSize', 18);
plot([x3(1) x3(2) x4(1)], [y3(1) y3(2) y4(1)], 'ks', 'MarkerSize', 18);
%text
text(2.5,8,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(2.8,1.5,'$\omega_2$','interpreter','latex', 'FontSize', 20);
% text(0.3,3.3,'$\displaystyle g_1\left( x \right)$','interpreter','latex', 'FontSize', 15);
% text(0.3,2.3,'$\displaystyle g_2\left( x \right)$','interpreter','latex', 'FontSize', 15);
% text(0.3,1.3,'$\displaystyle g_3\left( x \right)$','interpreter','latex', 'FontSize', 15);
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([-0.5, 7]);
ylim([0, 10]);
hold off

%Ulozeni
print(h, '-dpdf', 'SVM03');

%% SVM - KernelTrick DATA
ax = 0.5;
bx = 6.5;
x = (bx-ax).*rand(100,1) + ax;
ay = 0.5;
by = 9.5;
y = (by-ay).*rand(100,1) + ay;
r = 2;
ang=0:0.01:2*pi;
omg1 = 0;
omg2 = 0;

h = figure;
subplot(1,2,1)
hold on; 
xp=r*cos(ang);
yp=r*sin(ang);
plot(3.5+xp,5+yp,'k');
for i=1:length(x)
    if (sqrt((3.5-x(i))^2+(5-y(i))^2) > 2.5)
        plot(x(i), y(i), 'x', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
        omg2 = omg2 + 1;
    elseif (sqrt((3.5-x(i))^2+(5-y(i))^2) < 1.5)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');
        omg1 = omg1 + 1;
    end
end
%text
text(3.5,6.5,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(3.5,9.5,'$\omega_2$','interpreter','latex', 'FontSize', 20);
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([-0.5, 7]);
ylim([0, 10]);
hold off;

%Ulozeni
%print(gcf, '-dpdf', 'KT01');

%%
xx = (bx-ax).*rand(1000,1) + ax;
yy = (bx-ax).*rand(1000,1) + ax;
zz1 = ones(1,omg1)*10;
%zz2 = ones(1,omg2)*10;
zz2 = (10-6).*rand(omg2,1) + 6;
poc1 = 0;
poc2 = 0;
for i=1:length(xx)
    if ((5/3)*xx(i)+yy(i)<4 && poc1<omg1)
        poc1 = poc1 + 1;
        xx1(poc1) = xx(i);
        yy1(poc1) = yy(i);
    end
    if ((5/3)*xx(i)+yy(i)>6 && poc2<omg2)
        poc2 = poc2 + 1;
        xx2(poc2) = xx(i);
        yy2(poc2) = yy(i);
    end
end
%h = figure;
subplot(1,2,2)
plot3(xx1, zz1, yy1, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');
hold on;
plot3(xx2, zz2, yy2, 'x', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
line([3 0],[10 10],[0 5], 'Color', 'k','LineWidth', 1, 'LineStyle', '--');
line([0 0],[10 0],[5 5], 'Color', 'k','LineWidth', 1, 'LineStyle', ':');
line([0 3],[0 0],[5 0], 'Color', 'k','LineWidth', 1, 'LineStyle', '--');
line([3 3],[0 10],[0 0], 'Color', 'k','LineWidth', 1, 'LineStyle', ':');
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
zlabel('x_3', 'FontSize', 12);
xlim([-0.5, 7]);
ylim([0, 10]);
zlim([0, 10]);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gca,'ZTick',[]);
hold off;

%print(gcf, '-dpdf', 'KT01');
%% porovnani img 01
%DATA
ax = 0.5;
bx = 6.5;
x = (bx-ax).*rand(250,1) + ax;
ay = 0.5;
by = 9.5;
y = (by-ay).*rand(250,1) + ay;
r = 1.7;
ang=0:0.01:2*pi;
omg1 = 0;
omg2 = 0;
omg3 = 0;

h = figure;
hold on; 
xp=r*cos(ang);
yp=r*sin(ang);
plot(2+xp,2+yp,'k');
plot(5.5+xp,4+yp,'k');
plot(4+xp,8+yp,'k');
for i=1:length(x)
    if (sqrt((2-x(i))^2+(2-y(i))^2) < 1.5)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
    elseif (sqrt((5.5-x(i))^2+(4-y(i))^2) < 1.5)
        plot(x(i), y(i), 's', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'b', 'MarkerFaceColor','b');
    elseif (sqrt((4-x(i))^2+(8-y(i))^2) < 1.5)
        plot(x(i), y(i), '^', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');    
    end
end
%text
text(2,4,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(5.5,6,'$\omega_2$','interpreter','latex', 'FontSize', 20);
text(4,10,'$\omega_2$','interpreter','latex', 'FontSize', 20);
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([-0.5, 7.5]);
ylim([0, 10.5]);
hold off;

%Ulozeni
%print(gcf, '-dpdf', 'porov01');

%% porovnani img 02
ax = -0.5;
bx = 6.5;
x = (bx-ax).*rand(1550,1) + ax;
ay = 0.5;
by = 9.5;
y = (by-ay).*rand(1550,1) + ay;
r = 1.4;
r2 = 1;
ang=0:0.01:2*pi;
x1 = [3.1:0.1:3.9];
y1 = [3.6,4.1,3.05,3.65,5.4,6.8,6.3,5.3,4.8];

h = figure;
hold on; 
xp=r*cos(ang);
yp=r*sin(ang);
xp2=r2*cos(ang);
yp2=r2*sin(ang);
plot(1+xp,4.5+yp,'k');
plot(5.5+xp2,4.5+yp2,'k');
for i=1:length(x)
    if (sqrt((1-x(i))^2+(4.5-y(i))^2) < 1.2)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k');
    elseif ((sqrt((5.5-x(i))^2+(4.5-y(i))^2) > 0.9) && (sqrt((5.5-x(i))^2+(4.5-y(i))^2) < 1.1))
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r');
    end
end
plot(x1, y1, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'b', 'MarkerFaceColor','b');
line([3 4],[2.5 7], 'Color', 'k');
%text
text(1,2,'$a)$','interpreter','latex', 'FontSize', 20);
text(3.5,2,'$b)$','interpreter','latex', 'FontSize', 20);
text(5.5,2,'$c)$','interpreter','latex', 'FontSize', 20);
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([-0.5, 7]);
ylim([0, 10]);
hold off;



%Ulozeni
print(gcf, '-dpdf', 'porov02');

%% DT img 01 - part 01
%DATA
a = 0.1;
b = 0.9;
x = (b-a).*rand(500,1) + a;
y = (b-a).*rand(500,1) + a;

h = figure;
subplot(1,2,1)
hold on;
for i=1:length(x)
    if (x(i)<0.2)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
    end
    if (x(i)>0.3 && x(i)<0.7 && y(i)<0.45)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
    end
    if (x(i)>0.3 && x(i)<0.45 && y(i)>0.55)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
    end
    if (x(i)>0.8 && y(i)<0.45)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
    end
    if (x(i)>0.55 && y(i)>0.55 && y(i)<0.65)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
    end
    if (x(i)>0.55 && y(i)>0.8)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
    end
end
%lines
line([0.25 0.25],[0 1], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.75 0.75],[0 0.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.5 0.5],[0.5 1], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.25 1],[0.5 0.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.5 1],[0.75 0.75], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
%text
text(0.12,0.05,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(0.48,0.05,'$\omega_3$','interpreter','latex', 'FontSize', 20);
text(0.85,0.05,'$\omega_4$','interpreter','latex', 'FontSize', 20);
text(0.35,0.95,'$\omega_2$','interpreter','latex', 'FontSize', 20);
text(0.7,0.95,'$\omega_3$','interpreter','latex', 'FontSize', 20);
text(0.7,0.7,'$\omega_1$','interpreter','latex', 'FontSize', 20);
%axsis
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[0.25,0.5,0.75]);
set(gca,'YTick',[0.25,0.5,0.75]);
xlim([0, 1]);
ylim([0, 1]);
hold off;

%% DT img 01 - part 02

r=0.5;
ang=0:0.01:2*pi;
xp=r*cos(ang);
yp=r*sin(ang);

%h = figure;
subplot(1,2,2)
hold on;
%nodes
plot(2+xp,9+yp,'k');
plot(4+xp,7+yp,'k');
plot(2+xp,5+yp,'k');
plot(6+xp,5+yp,'k');
plot(8+xp,3+yp,'k');
%terminals
line([0.25 1.35],[7.5 7.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.25 1.35],[6.5 6.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.25 0.25],[6.5 7.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([1.35 1.35],[6.5 7.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');

line([0.25 1.35],[3.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.25 1.35],[2.5 2.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([0.25 0.25],[2.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([1.35 1.35],[2.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');

line([2.75 3.85],[3.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([2.75 3.85],[2.5 2.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([2.75 2.75],[2.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([3.85 3.85],[2.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');

line([4.25 5.35],[3.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([4.25 5.35],[2.5 2.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([4.25 4.25],[2.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([5.35 5.35],[2.5 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');

line([6.25 7.35],[1.5 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([6.25 7.35],[0.5 0.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([6.25 6.25],[0.5 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([7.35 7.35],[0.5 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');

line([8.75 9.85],[1.5 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([8.75 9.85],[0.5 0.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([8.75 8.75],[0.5 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([9.85 9.85],[0.5 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
%paths
line([2.365 3.722],[8.658 7.416], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([3.7 2.274],[6.6 5.418], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([4.302 5.726],[6.601 5.418], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([6.358 7.702],[4.651 3.401], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([5.644 4.75],[4.649 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([2.358 3.25],[4.651 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([1.644 0.75],[4.649 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([8.351 9.25],[2.644 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([7.651 6.75],[2.642 1.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([1.637 0.75],[8.656 7.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');

%text
text(1.6,9,'$t_0$','interpreter','latex', 'FontSize', 20);
text(3.6,7,'$t_1$','interpreter','latex', 'FontSize', 20);
text(1.6,5,'$t_2$','interpreter','latex', 'FontSize', 20);
text(5.6,5,'$t_3$','interpreter','latex', 'FontSize', 20);
text(7.6,3,'$t_4$','interpreter','latex', 'FontSize', 20);

text(0.3,7,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(0.3,3,'$\omega_3$','interpreter','latex', 'FontSize', 20);
text(2.8,3,'$\omega_4$','interpreter','latex', 'FontSize', 20);
text(4.3,3,'$\omega_2$','interpreter','latex', 'FontSize', 20);
text(6.3,1,'$\omega_1$','interpreter','latex', 'FontSize', 20);
text(8.8,1,'$\omega_3$','interpreter','latex', 'FontSize', 20);

text(2.6,9,'$x_1\leq\frac{1}{4}$','interpreter','latex', 'FontSize', 14);
text(4.6,7,'$x_2\leq\frac{1}{2}$','interpreter','latex', 'FontSize', 14);
text(2.6,5,'$x_1\leq\frac{3}{4}$','interpreter','latex', 'FontSize', 14);
text(6.6,5,'$x_1\leq\frac{1}{2}$','interpreter','latex', 'FontSize', 14);
text(8.6,3,'$x_2\leq\frac{3}{4}$','interpreter','latex', 'FontSize', 14);

text(0.05,8.3,'ANO', 'FontSize', 10);
text(2.9,8.3,'NE', 'FontSize', 10);
text(2.05,6.3,'ANO', 'FontSize', 10);
text(4.9,6.3,'NE', 'FontSize', 10);
text(0.05,4.3,'ANO', 'FontSize', 10);
text(2.8,4.3,'NE', 'FontSize', 10);
text(4.05,4.3,'ANO', 'FontSize', 10);
text(6.9,4.3,'NE', 'FontSize', 10);
text(6.05,2.3,'ANO', 'FontSize', 10);
text(8.8,2.3,'NE', 'FontSize', 10);

line([0 11.05],[10 10], 'Color', 'k','LineWidth', 1,'LineStyle', '-');
line([11.05 11.05],[0 10], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
%axsis
%xlabel('x_1', 'FontSize', 12);
%ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([0, 11.05]);
ylim([0, 10]);
hold off;

%Ulozeni
%print(gcf, '-dpdf', 'DT01');

%% DT img 02
%DATA
a = 0.1;
b = 9.9;
x = (b-a).*rand(500,1) + a;
y = (b-a).*rand(500,1) + a;

r=0.5;
ang=0:0.01:2*pi;
xp=r*cos(ang);
yp=r*sin(ang);

h = figure;
hold on;
%nodes
plot(5+xp,8+yp,'k');
plot(4+xp,3+yp,'k');
plot(6+xp,3+yp,'k');
count = 0;

for i=1:length(x)    
    if (sqrt((7.5-x(i))^2+(8-y(i))^2) < 1.5)
        plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r'); 
        count = count+1;
    end
end
for i=1:length(x)
    if (sqrt((2-x(i))^2+(3-y(i))^2) < 1.2)
        if (count > 0)
            plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
        end
        count = count-1;
    elseif (sqrt((8-x(i))^2+(3-y(i))^2) < 1.2)
        if (count > 0)
            plot(x(i), y(i), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','r');
        end
        count = count-1;
    end
end
%paths
line([5.103 6],[7.511 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
line([4.895 4],[7.511 3.5], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
%text
text(4.9,8,'$t$','interpreter','latex', 'FontSize', 20);
text(3.7,3,'$t_A$','interpreter','latex', 'FontSize', 20);
text(5.7,3,'$t_N$','interpreter','latex', 'FontSize', 20);

text(7.5,5.9,'$X_t$','interpreter','latex', 'FontSize', 20);
text(2,1.2,'$X_{tA}$','interpreter','latex', 'FontSize', 20);
text(8,1.2,'$X_{tN}$','interpreter','latex', 'FontSize', 20);

line([0 10],[10 10], 'Color', 'k','LineWidth', 1,'LineStyle', '-');
line([10 10],[0.7 10], 'Color', 'k','LineWidth', 1, 'LineStyle', '-');
%axsis
%xlabel('x_1', 'FontSize', 12);
%ylabel('x_2', 'FontSize', 12);
set(gca,'XTick',[]);
set(gca,'YTick',[]);
xlim([0, 10]);
ylim([0.7, 10]);
hold off;

%Ulozeni
print(gcf, '-dpdf', 'DT02');
