clear
close all
cd /Users/jj/GoogleDrive/Course/GPUProgramming/proj2/TimingMatrixMult
t_cpu = load('CPU/time.txt');
t_naiveGPU_block64 = load('naiveGPU/block64/time.txt');
t_naiveGPU_block32 = load('naiveGPU/block32/time.txt');
t_tile32GPU = load('tiledGPU/tile32/time.txt');
t_tile64GPU = load('tiledGPU/tile64/time.txt');

figure;
plot(t_cpu(:,1),t_cpu(:,2), '-ys');
hold on 
plot(t_naiveGPU_block32(:,1), t_naiveGPU_block32(:,2),'-ro');
hold on 
plot(t_naiveGPU_block64(:,1), t_naiveGPU_block64(:,2),'-bx');
hold on 
plot(t_tile32GPU(:,1),t_tile32GPU(:,2),'--g+');
hold on
plot(t_tile64GPU(:,1),t_tile64GPU(:,2),'--k*');
set(gca,'YScale','log');
set(gca,'XScale','log');
legend('CPU serial','GPU naive block32', 'GPU naive block64', 'GPU tile 32 block64', 'GPU tile 64 block64','Location','northwest');
xlabel('Matrix dimensions');
ylabel('Time consumed (s)');
title('Time comparisons');
print('TimeComparisons.png','-dpng');

figure();
plot(t_naiveGPU_block32(:,1), t_cpu(:,2)./t_naiveGPU_block32(:,2),'-ro');
hold on 
plot(t_naiveGPU_block64(:,1), t_cpu(:,2)./t_naiveGPU_block64(:,2),'-bx');
hold on 
plot(t_tile32GPU(:,1),t_cpu(:,2)./t_tile32GPU(:,2),'--g+');
hold on
plot(t_tile64GPU(:,1),t_cpu(:,2)./t_tile64GPU(:,2),'--k*');
set(gca,'YScale','log');
set(gca,'XScale','log');
legend('GPU naive block32', 'GPU naive block64', 'GPU tile 32 block64', 'GPU tile 64 block64','Location','northwest');
xlabel('Matrix dimensions');
ylabel('Speed up');
title('Speedup comparisons');
print('SpeedupComparisons.png','-dpng');
