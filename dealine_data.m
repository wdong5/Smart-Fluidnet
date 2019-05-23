load('deadline_1.txt');
load('deadline_2.txt')
load('deadline_4.txt')
load('deadline_1_15.txt')
load('deadline_2_15.txt')
load('deadline_4_15.txt')

with(:,1) = deadline_1(1:40,2)
with(:,2) = deadline_2(1:40,2)
with(:,3) = deadline_4(1:40,2)

without(:,1) = deadline_1_15(1:40,2)
without(:,2) = deadline_2_15(1:40,2)
without(:,3) = deadline_4_15(1:40,2)

with_avg = mean(with, 1)
without_avg = mean(without, 1)

speedup = with_avg/without_avg