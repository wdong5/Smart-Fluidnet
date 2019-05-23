%image quality
load('baseline/statistics_default_all.txt')

x_1 = [1:510];
y_1 = statistics_default_all(2:511,2)';
figure(1);
plot(x_1,y_1);
legend('default-imageQuality');
grid on


%divNorm 
load('baseline/default_all_Stats.bin.mat')
default_normDiv = normDiv;

x_2 = [1:511];
y_2 = default_normDiv(2:512,1)';
figure(2);
plot(x_2,y_2);
legend('default-normDiv');
grid on

%%cum sum of the divNorm

cum_sum_default_normDiv = cumsum(default_normDiv(2:end-1,1));

x_3 = [1:510];
y_3 = cum_sum_default_normDiv(:,1);
figure(3);
plot(x_3,y_3);
legend('default-cunSum-normDiv');
grid on







