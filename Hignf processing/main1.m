clear all
close all
clc
warning off
%% ceemdan分解
% 加载信号
%signal=csvread('S_ddc_waitCPO.csv');
signal=csvread('S_real30.csv');

%{
%% 分解
addpath(genpath(pwd)) % 添加路径
D_num =5; 
IMF = decomposition_compilations(signal,D_num); % imf格式为：模态个数 x 数据长度

rmpath(genpath(pwd)) % 移除路径
%% 绘图-最后一个imf可视为残差

% plot_func(signal, IMF)

%% 二次分解 CPO-VMD分解
%% 参数设置
d = length(IMF(:,3840))
data=IMF(1,:) 
for i=2:d
    data=data + IMF(i,:);
end
len=length(data);
f=data(1:len);
%}

data = signal;
len=length(data);
f = data(1:len)

% alpha = 2000;        % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
% K = 4;              % 4 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 1e-7;
 
%% 普通VMD分解
%[u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
% 分解
[u1, u_hat1, omega1,curve,Target_pos] = WLVMD(f, tau, DC, init, tol);
%% 

figure %1
plot(curve,'linewidth',1.5);
title('收敛曲线')
xlabel('迭代次数')
ylabel('适应度值')
grid on

%分解
figure %2
subplot(size(u1,1)+1,1,1);
plot(f,'k');grid on;
title('原始数据');
for i = 1:size(u1,1)
    subplot(size(u1,1)+1,1,i+1);
    plot(u1(i,:),'k');
end

disp(['最优K值为：',num2str(Target_pos(2))])
disp(['最优alpha值为：',num2str(Target_pos(1))])
disp(['最优综合指标为：',num2str(min(curve))])
%% 计时结束
%% 频域图
[m,n]=size(u1);
imf=u1;
len=length(imf);%信号长度
fs=3840;%采样频率
% 采样时间
t = (0:len-1)/fs; 
figure %3
for i=1:m
subplot(m,1,i)
[cc,y_f]=hua_fft_1(imf(i,:),fs,1);
a1(i,:)=cc;
plot(y_f,cc,'k','LineWIdth',1.5);
% hua_fft_1(u(i,:),fs,1)
ylabel(['imf',num2str(i)]);
axis tight
end
xlabel('频率/Hz')
%% Hilbert边际谱 
figure %4
plot(y_f,a1,'LineWIdth',1);
title('Hilbert边际谱')
xlabel('频率/Hz')
ylabel('幅值')
%% 分解结果整合
%{

%写入同一个csv文件中
[numIMFs, len] = size(IMF); % 获取IMF的数量和信号长度
% 创建一个空矩阵，用于存储所有IMFs
% 初始时，矩阵有 len 行，但列数为0
allIMFsMatrix = zeros(len, 0);
% 循环遍历每个IMF并添加到矩阵中作为新的列
for i = 1:numIMFs
    % 将第i个IMF添加到矩阵的新的列中
    allIMFsMatrix(:, end+1) = IMF(i, :);
end
% 指定CSV文件名
csvFileName = 'CEE_IMFsd.csv';
csvwrite(csvFileName, allIMFsMatrix);


u=[IMF(d+1:end,:);imf];%更改过的参数2，3，4
plot_func(signal, u)

%写入同一个csv文件中
[numIMFs, len] = size(u); % 获取IMF的数量和信号长度

% 创建一个空矩阵，用于存储所有IMFs
% 初始时，矩阵有 len 行，但列数为0
allIMFsMatrix = zeros(len, 0);

% 循环遍历每个IMF并添加到矩阵中作为新的列
for i = 1:numIMFs
    % 将第i个IMF添加到矩阵的新的列中
    allIMFsMatrix(:, end+1) = u(i, :);
end

% 指定CSV文件名
csvFileName = 'All_IMFsd.csv';

% 将原始信号添加到 allIMFsMatrix 的第一列
%allIMFsMatrix = [signal, allIMFsMatrix];
csvwrite(csvFileName, allIMFsMatrix);
%}

%{
%写入文件
[numIMFs, len] = size(imf); % 获取IMF的数量和信号长度

% 创建一个文件夹来存储CSV文件，如果文件夹不存在则创建它
csvFolder = 'IMF_CSV_Files';
if ~exist(csvFolder, 'dir')
    mkdir(csvFolder);
end

% 循环遍历每个IMF并写入到单独的CSV文件中
for i = 1:numIMFs
    % 创建CSV文件的文件名，例如: 'IMF1.csv', 'IMF2.csv', ...
    csvFileName = fullfile(csvFolder, sprintf('IMF%d.csv', i));
    
    % 将当前IMF写入CSV文件
    % 假设IMF已经是列向量，如果不是，需要转置
    csvwrite(csvFileName, imf(i, :));
end

% 如果需要保存原始信号，也可以将其保存为CSV文件
% 将原始信号保存在名为 'Original_Signal.csv' 的文件中
originalSignalCSV = fullfile(csvFolder, 'Original_Signal.csv');
csvwrite(originalSignalCSV, signal);
%}

