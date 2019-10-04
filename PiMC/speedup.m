CPU=load('CPU/results.txt');
GPU=load('GPU/results.txt');
figure
subplot(1,2,1)
semilogx(CPU(:,1), CPU(:,3),'-rx');
hold on
semilogx(GPU(:,1), GPU(:,3)+GPU(:,4),'-bo');
xlabel('Sample Size')
ylabel('Time (ms)')
title('Time cost in computing Pi')
legend('CPU','GPU','Location','northwest')

subplot(1,2,2)
semilogx(CPU(:,1), CPU(:,3)./(GPU(:,3)+GPU(:,4)),'-bx');
xlabel('Sample Size')
ylabel('Speedup')
title('GPU Speedup over CPU in computing Pi')

print('TimeAnalysis_PiMC.pdf', '-dpdf')