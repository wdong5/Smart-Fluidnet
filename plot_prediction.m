load('test/statistics_5.txt');
load('test/predict_statistics_5.txt');


x_1 = [1:511];
y_1 = [statistics_5(1:511,2)];

figure(1);

plot(x_1,y_1,'color','b');
hold on % hold zhu 

x_2 = [predict_statistics_5(1:103,1)];
y_2 = [predict_statistics_5(1:103,2)];

scatter(x_2,y_2,1); %'MarkerFaceColor',[0 .7 .7]);
grid on
legend('ground truth','predicted', 'Location','northwest');
