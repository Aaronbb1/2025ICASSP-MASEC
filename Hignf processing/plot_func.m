function plot_func(signal,imf)
%% 图像美观需自行调整
% figure
% plot(signal)
% xlim([0 length(signal)])
% title('原始信号')
% 3d
figure
set(gcf,'unit','normalized','position',[0.2,0.3,0.5,0.45]);
x = 1:size(imf,1);
y = 1:size(imf,2);
z = imf(x,:);
[X,Y]=meshgrid(x,y);
plot3(X,Y,z)
grid on
xticklabel = {};
for ii=1:size(imf,1)
    
    xticklabel{ii}= ['IMF' num2str(ii)];
    
end
set(gca,'xtick',1:1:size(imf,1),'XTickLabel',xticklabel);
view(-20, 50); %视角
% 2d
figure
for i=1:size(imf,1)
    subplot(size(imf,1),1,i)
    plot(imf(i,:))
    
    ylabel(['IMF' num2str(i)]);
    
    xlim([0 length(signal)])
end

%% 频谱
%采样频率
fs=12800; %5120 等其他频率

figure
for i=1:size(imf,1)
    subplot(size(imf,1),1,i)
    y_f=calculate_fft(imf(i,:),fs);
    plot(y_f,'LineWIdth',1);
    ylabel(['IMF',num2str(i)]);
end


end