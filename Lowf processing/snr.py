import numpy as np

def calculate_snr(signal, noise):
    """
    计算信噪比（dB）
    """
    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
def reconstruct_signal_with_snr(signal, imfs, snr_threshold):
    """
    使用信噪比阈值重构信号
    """
    selected_imfs = []
    n = 0

    # 计算每个IMF的信噪比并选择小于阈值的IMF
    for imf in imfs:
        snr = calculate_snr(signal - imf, imf)
        n = n + 1
        if snr < snr_threshold:
            selected_imfs.append(imf)
            #print(n)

    # 重构信号
    reconstructed_signal = np.sum(selected_imfs, axis=0)
    return reconstructed_signal

def mse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def TMAPE(actual, predicted):
    # 检查实际值中是否有零，以避免除以零的错误
    if np.any(actual == 0) and np.any(predicted == 0):
        raise ValueError("TMAPE cannot be calculated when actual and predicted values are both zero.")
    # 计算绝对百分比误差
    absolute_percentage_error = np.abs((actual - predicted) / actual)
    # 计算平均值
    mean_absolute_percentage_error = np.mean(absolute_percentage_error)
    return mean_absolute_percentage_error

def SMAPE(y_true, y_pred):
    if np.any(y_true == 0) and np.any(y_pred == 0):
        raise ValueError("TMAPE cannot be calculated when actual and predicted values are both zero.")
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))