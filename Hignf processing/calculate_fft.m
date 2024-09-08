function [cc,y_f]=calculate_fft(y,fs)

nfft= 2^nextpow2(length(y));%找出大于y的个数的最大的2的指数值（自动进算最佳FFT步长nfft）
%nfft=1024;%人为设置FFT的步长nfft
y=y-mean(y);%去除直流分量
y_ft=fft(y,nfft);%对y信号进行DFT，得到频率的幅值分布

y_f=fs*(0:nfft/2-1)/nfft;%T变换后对应的频率的序列


cc=2*abs(y_ft(1:nfft/2))/length(y);



end

