% body
P0 = [1,1];
P1 = [2,3];
P2 = [2.5,4];
P3 = [3,3.2];
P4 = [4.5,1.5];
P5 = [2.7,1.2];

P = [P0;P1;P2;P3;P4;P5];

% fergassonovy kubiky
F0 = @(t) (2*t.^3 - 3*t.^2 + 1);
F1 = @(t) (-2*t.^3 + 3*t.^2);
F2 = @(t) (t.^3 - 2*t.^2 + t);
F3 = @(t) (t.^3 - t.^2);

n = 6;                  %pocet bodu
A = zeros(n,n);
b1 = zeros(n,1);
b2 = zeros(n,1);

% koeficienty pro vypocet smerovych vektoru
A = A + diag(4*ones(n,1),0);
A = A + diag(ones(n-1,1),1);
A = A + diag(ones(n-1,1),-1);
% aby bylo spojite na konci
A(1,n) = 1;
A(n,1) = 1;

% prave strany pro derivace
for i = 2:n-1
    b1(i) = 3*(P(i+1,1) - P(i-1,1));
    b2(i) = 3*(P(i+1,2) - P(i-1,2));
end
% aby bylo spojite na konci
b1(1) = 3*(P1(1) - P5(1));
b2(1) = 3*(P1(2) - P5(2));
b1(n) = 3*(P0(1) - P4(1));
b2(n) = 3*(P0(2) - P4(2));

x = A\b1;
y = A\b2;


figure(1)
plot(P(:,1),P(:,2),'.')
hold on

for i = 1:5
    tt = 0:0.01:1;
    kubika_x = F0(tt)*P(i,1) + F1(tt)*P(i+1,1) + F2(tt)*x(i) + F3(tt)*x(i+1);%x
    kubika_y = F0(tt)*P(i,2) + F1(tt)*P(i+1,2) + F2(tt)*y(i) + F3(tt)*y(i+1);%y
    plot(kubika_x,kubika_y)
end
tt = 0:0.01:1;
kubika_x = F0(tt)*P(6,1) + F1(tt)*P(1,1) + F2(tt)*x(6) + F3(tt)*x(1);
kubika_y = F0(tt)*P(6,2) + F1(tt)*P(1,2) + F2(tt)*y(6) + F3(tt)*y(1);
plot(kubika_x, kubika_y)

