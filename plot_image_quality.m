
load('convnet_images/statistics_0.txt')
load('convnet_images/statistics_32.txt')
load('convnet_images/statistics_64.txt')
load('convnet_images/statistics_96.txt')
%normDiv_sub196_default = normDiv;


x = [1:127];
y = [statistics_0(:,2)';
    statistics_32(:,2)';
    statistics_64(:,2)';
    statistics_96(:,2)'];

figure(1);
plot(x,y);
%legend('default all the time','pcg-32 default','pcg-64 default','pcg-96 default');
grid on
