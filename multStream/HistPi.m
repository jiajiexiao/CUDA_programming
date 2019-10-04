clear
close all
data=load('hist.txt');
%data=load('GPU/hist.txt');
figure
title('Distribution of digits in Pi')

for n=1:6
subplot(2,3,n)
labels ={'0','1','2','3','4','5','6','7','8','9'};
pie(data(n,2:end)/abs(data(n,1)), labels);
    titlename=strcat('First   ', num2str(data(n,1)),' digits');
    title(titlename);
end

print('Hist_Pi.pdf', '-dpdf')