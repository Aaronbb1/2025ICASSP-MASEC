import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snr import reconstruct_signal_with_snr, calculate_snr, mse
from vmdpy import VMD,find_peaks
from detrend import lin_window_ovrlp, exp_detrend, poly_detrend, poly_detrend1

# 读取 MATLAB 输出的 CSV 文件
data = pd.read_csv('S_real.csv', header=None)
#data_noise = pd.read_csv('S_noise.csv', header=None)
#data_noise = data_noise.values.flatten()
data_0 = pd.read_csv('S0.csv', header=None)
data_0 = data_0.values.flatten()
fs = 3840; #采样频率
T = 1/fs; #采样周期
t = np.arange(T, 1 + T, T)
f0 = 60;
start_index=500
end_index=3500
freqs = 2*np.pi*(t-0.5-fs)/(fs)

data = data.values.flatten()
r_data = data[3840-384:3840]
l_data = data[0:384]
data_ep = np.concatenate((data, r_data), axis=0)
data_ep = np.concatenate((l_data, data_ep), axis=0)
t_ep = np.arange(T, 1+0.2+T, T)
#data_flipped = data[::-1]

'''
#每个高次多项式单独拟合结果图像
plt.plot(t+0.1,5*np.exp(-5*t) , c='k', linewidth=0.8, alpha=1 )
all_detrended = []
for i in range(1,9):
    y_detrend = poly_detrend1(t_ep, data_ep, degree=i, length=76, stopper=50, n_ovrlp=5)
    all_detrended.append(data_ep - y_detrend)
    DC_detrend_ep = np.mean(all_detrended, axis=0)
    DC_detrend = DC_detrend_ep[384:4224]
    plt.plot(t_ep, DC_detrend_ep, label=f'degree_{i}', linewidth=f'{i*0.1}', alpha=1)
plt.legend(loc='upper center', frameon=False)
plt.savefig('fit0.png', dpi=3000)
plt.show()
'''
#===============================================
# 定义参数的搜索范围
min_degree = 1
max_degree = 10
# 初始化最佳 MSE 和最佳参数
best_mse = float('inf')
best_params = (0, 0)
# 遍历所有可能的 degree1 和 degree2 的组合
for degree1 in range(min_degree, max_degree):
    for degree2 in range(degree1 + 1, max_degree + 1):  # degree2 必须大于 degree1
        # 评估当前参数组合的性能
        all_detrended = []
        for i in range(degree1, degree2):
            y_detrend = poly_detrend1(t_ep, data_ep, degree=i, length=76, stopper=50, n_ovrlp=5)
            all_detrended.append(data_ep - y_detrend)
        DC_detrend_ep = np.mean(all_detrended, axis=0)
        DC_detrend = DC_detrend_ep[384:4224]
        mse_value = mse(DC_detrend,5*np.exp(-5*t))
    if mse_value < best_mse:
        best_mse = mse_value
        best_params = (degree1, degree2)

print(f"Best parameters: degree1={best_params[0]}, degree2={best_params[1]}, MSE={best_mse}")
#===============================================
degree1=best_params[0]
degree2=best_params[1]
plt.plot(t+0.1,5*np.exp(-5*t) , c='k', linewidth=0.5, alpha=1 )
all_detrended = []
for i in range(degree1,degree2):
    y_detrend = poly_detrend1(t_ep, data_ep, degree=i, length=76, stopper=50, n_ovrlp=5)
    all_detrended.append(data_ep - y_detrend)
DC_detrend_ep = np.mean(all_detrended, axis=0)
DC_detrend = DC_detrend_ep[384:4224]
plt.plot(t_ep, DC_detrend_ep, label='mean_all', linewidth=0.5, alpha=1, linestyle='--')
plt.plot(t+0.1, DC_detrend, label='mean', linewidth=0.5, alpha=1)
plt.legend(loc='upper center', frameon=False)
#plt.savefig('fit.png', dpi=3000)
plt.show()

mse_value = mse(data, data_0)
print("MSE_data vs data_0:", mse_value)

data = data - DC_detrend

#data数据去除直流衰减分量完成
# some sample parameters for VMD
alpha = 2050       # 数据保真度约束的平衡参数，用于调整数据保真度和模式平滑度之间的权衡。
tau = 0            # 双重上升算法的时间步长，如果设置为0，则不对噪声进行严格处理。
K = 17
DC = 0            # DC: 布尔值，如果为True，则将第一个模式固定在直流（0频率）。
init = 1           # 初始化方式：0：所有中心频率 omega 从0开始。1：所有 omega 均匀分布。2：所有 omega 随机初始化。
tol = 1e-6         #收敛性容差，通常设置在1e-6左右，用于判断算法是否收敛。

# Run actual VMD code
u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
"""
#VMD分解的子集结果
plt.figure(figsize=(10, 8))  # 创建一个新图形，并设置图形大小
plt.subplot(K+1, 1, 1)
plt.plot(data[start_index:end_index], linewidth=1)
for i in range(K):  # 遍历每个模式
    plt.subplot(K+1, 1, i + 2)  # 创建子图，K 行 1 列，当前是第 i+1 个子图
    plt.plot(u[i,:][start_index:end_index], linewidth=1)  # 绘制第 i 个模式
    plt.ylabel(f'IMF{i}')  # 设置 y 轴标签为 'IMF' 加上模式编号
    plt.tight_layout()  # 紧凑布局以避免子图间的重叠
plt.savefig('VMD.png', dpi=3000)
plt.show()  # 显示图形
"""

'''
#分成多页绘制IMF子图
num_pages = (K + 5) // 6  # 向上取整
for page in range(num_pages):
    plt.figure(figsize=(10, 8))  # 创建一个新图形，并设置图形大小
    start = 6 * page + 1  # 计算当前页的起始索引
    end = min(6 * (page + 1), K)  # 计算当前页的结束索引

    # 绘制模式
    for i in range(end - start + 1):  # 遍历每个模式
        plt.subplot(6, 1, i + 1)  # 创建子图，6 行 1 列，当前是第 i+1 个子图
        plt.plot(u[page * 6 + i, :], linewidth=1)  # 绘制第 i 个模式
        plt.ylabel(f'IMF{page * 6 + i + 1}')  # 设置 y 轴标签为 'IMF' 加上模式编号

    plt.tight_layout()  # 紧凑布局以避免子图间的重叠
    #plt.savefig(f'VMD_page_{page + 1}.png', dpi=3000)
    #plt.show()
'''

IMFs = []
for i in range(K):
    IMFs.append(u.T[:, i])  # 将每个 IMF 添加到列表的末尾
'''
#对每个IMF进行FFT的图像表示
fft_data = np.fft.fft(data[start_index:end_index], n=None)
fft_IMFs = [np.fft.fft(IMF[start_index:end_index], n=None) for IMF in IMFs]
# 计算频率轴的刻度
fft_length = len(data[start_index:end_index])
fft_frequency = np.fft.fftfreq(fft_length, d=(t[1]-t[0]))

# 由于FFT结果是对称的，我们只需要一半的频率分量
half_length = fft_length // 2
fft_frequency_positive = fft_frequency[:half_length]
# 绘制原始信号的 FFT（正频率部分）并标记所有极大值点
plt.figure()
plt.subplot(K + 2, 1, 1)
fft_original = np.abs(np.fft.fft(data[start_index:end_index])[:half_length])
peaks_indices = find_peaks(fft_original)
plt.plot(fft_frequency_positive, fft_original, linewidth=0.1, alpha=1)
plt.scatter(fft_frequency_positive[peaks_indices], fft_original[peaks_indices], s=0.5, color='red')  # 标记所有极大值点，s=50 控制点的大小
plt.title("Ori-Signal FFT")
print(fft_frequency_positive[peaks_indices])

# 绘制每个 IMF 的 FFT（正频率部分）并标记所有极大值点
for i, (IMF, fft_IMF) in enumerate(zip(IMFs, fft_IMFs), start=1):
    plt.subplot(K + 1, 1, i + 1)
    fft_imf = np.abs(fft_IMF[:half_length])
    peaks_indices = find_peaks(fft_imf)
    plt.plot(fft_frequency_positive, fft_imf, linewidth=0.1, alpha=1)
    plt.scatter(fft_frequency_positive[peaks_indices], fft_imf[peaks_indices],
                s=0.5, color='red')  # 标记所有极大值点
    plt.title(f'IMF {i} FFT')
    #print(f'IMF {i} ',fft_frequency_positive[peaks_indices])
#plt.savefig('VMD_fft.png', dpi=3000)
#plt.show()
'''
#找到使mse最小的IMF组成，其中默认已知data_0
min_mse = float('inf')  # 设置为无穷大
k_fit = None
mse_IMF = []
for k in range(1, K + 1):
    S_rec = np.zeros_like(IMFs[0])
    for i in range(k):
        S_rec = S_rec + IMFs[i]
    mse_value = mse(S_rec, data_0)
    mse_IMF.append(mse_value)

    if mse_value < min_mse:
        min_mse = mse_value
        k_fit = k
    #print(f"MSE {k}:  ", mse_value)
print(f"The minimum MSE is {min_mse:.4f} with {k_fit} IMFs.")
mse_value1 = mse(data, data_0)
print("MSE_data vs data_0:", mse_value1)
mse_value2 = mse(S_rec, data)
print("MSE_data vs S_rec:", mse_value2)

sig = np.empty(len(IMFs[0]))
plt.figure(figsize=(10, 6))  # 设置图像大小
for i in range(K):
    sig = sig + IMFs[i]
    print(f"{i+1}",calculate_snr(sig, data - sig))
    if i <= 6:
        color = ('lightcoral', 0.5)
    else:
        color = ('lightgreen', 0.5)
    plt.bar(i+1, calculate_snr(sig, data - sig) , color=color)
    plt.text(i + 1, calculate_snr(sig, data - sig), f'{calculate_snr(sig, data - sig):.2f}', ha='center', va='bottom')

plt.bar(0, 0, color=('lightcoral', 0.5), label='IMFs main')  # 假设前6个IMF用lightcoral颜色
plt.bar(7, 0, color=('lightgreen', 0.5), label='IMFs noise')  # 假设7到K个IMF用lightgreen颜色
plt.plot(range(5,len(mse_IMF)+1), mse_IMF[4:len(mse_IMF)], color = 'b', marker='o', markersize=4, markeredgecolor='r',label='mse')
plt.xlabel('IMF Number')
plt.ylabel('SNR')
plt.title('Signal to Noise Ratio for Each IMF')
plt.xticks(range(1, 18))  # 设置x轴刻度显示1到17的整数
plt.legend()
#plt.savefig('choose_IMF.png', dpi=3000)
plt.show()

# 信噪比阈值
snr_threshold = 28  # dB
# 重构信号
reconstructed_signal = reconstruct_signal_with_snr(data, IMFs, snr_threshold)
recon_snr = calculate_snr(reconstructed_signal, data - reconstructed_signal)
print(f"Reconstructed Signal SNR: {recon_snr:.2f} dB")

'''
#方差贡献率方法选择重建IMF的索引范围
variances = np.array([np.var(imf) for imf in IMFs])
# 计算方差贡献率
variance_contributions = variances / np.sum(variances)
# 绘制方差贡献率的条形图
plt.subplot()
plt.plot()
plt.bar(range(len(variance_contributions)), variance_contributions, color='skyblue')
plt.xlabel('IMF Index')
plt.ylabel('Variance Contribution')
plt.title('Variance Contributions of IMFs')
plt.xticks(range(len(variance_contributions)), [f'IMF {i+1}' for i in range(len(variance_contributions))])
plt.tight_layout()
plt.show()
'''

S_rec = np.empty(len(IMFs[0]))
for i in range(k_fit):
    S_rec= S_rec + IMFs[i]
with open('S_rec.csv', 'w', newline='') as file:
    # 创建一个csv写入器
    writer = csv.writer(file)
    # 将数组的每个元素写入CSV文件
    for element in S_rec:
        writer.writerow([element])
data_rec = pd.read_csv('S_rec.csv', header=None)
data_rec = data_rec.values.flatten()
print(data_rec.shape)
