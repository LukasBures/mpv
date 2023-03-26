

clear all;
close all;
clc;



signal = sin(0:0.1:4*pi);
signal = signal + rand(1, size(signal, 2));

idx = non_max_sup_1d(signal, 7, -1);



figure;
plot(cumsum(ones(length(signal))), signal, 'b');
hold on;

for i = 1:1:length(idx)
    if idx(i) ~= -1
        plot(i, signal(1, i), 'ro');
    end
end
    

