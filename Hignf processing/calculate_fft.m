function [cc,y_f]=calculate_fft(y,fs)

nfft= 2^nextpow2(length(y));%�ҳ�����y�ĸ���������2��ָ��ֵ���Զ��������FFT����nfft��
%nfft=1024;%��Ϊ����FFT�Ĳ���nfft
y=y-mean(y);%ȥ��ֱ������
y_ft=fft(y,nfft);%��y�źŽ���DFT���õ�Ƶ�ʵķ�ֵ�ֲ�

y_f=fs*(0:nfft/2-1)/nfft;%T�任���Ӧ��Ƶ�ʵ�����


cc=2*abs(y_ft(1:nfft/2))/length(y);



end

