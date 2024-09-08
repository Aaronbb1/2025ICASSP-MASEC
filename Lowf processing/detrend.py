import warnings
import numpy as np
from scipy.optimize import curve_fit, least_squares


def lin_window_ovrlp(time,y,length=3,stopper=3,n_ovrlp=2):
    """
    Windowed linear detrend function with optional window overlap
    
    Parameters
    ----------
    time : N x 1 numpy array
        Sample times.
    y : N x 1 numpy array
        Sample values.
    length : int
    stopper : int 
        minimum number of samples within each window needed for detrending
    n_ovrlp : int
        number of window overlaps relative to the defined window length
        
    Returns
        -------
        y.detrend : array_like
            estimated amplitudes of the sinusoids.
    
    Notes
    -----
    A windowed linear detrend function with optional window overlap for pre-processing of non-uniformly sampled data.
    The reg_times array is extended by value of "length" in both directions to improve averaging and window overlap at boundaries. High overlap values in combination with high
    The "stopper" values will cause reducion in window numbers at time array boundaries.   
    """
    x = np.array(time).flatten()
    y = np.array(y).flatten()
    y_detr      = np.zeros(shape=(y.shape[0]))
    counter     = np.zeros(shape=(y.shape[0]))
    A = np.vstack([x, np.ones(len(x))]).T
    #num = 0 # counter to check how many windows are sampled   
    interval    = length/(n_ovrlp+1) # step_size interval with overlap 
    # create regular sampled array along t with step-size = interval.         
    reg_times   = np.arange(x[0]-(x[1]-x[0])-length,x[-1]+length, interval)
    # extract indices for each interval
    idx         = [np.where((x > tt-(length/2)) & (x <= tt+(length/2)))[0] for tt in reg_times]  
    # exclude samples without values (np.nan) from linear detrend
    idx         = [i[~np.isnan(y[i])] for i in idx]
    # only detrend intervals that meet the stopper criteria
    idx         = [x for x in idx if len(x) >= stopper]
    for i in idx:        
        # find linear regression line for interval
        coe = np.linalg.lstsq(A[i],y[i],rcond=None)[0]
        # and subtract off data to detrend
        detrend = y[i] - (coe[0]*x[i] + coe[1])
        # add detrended values to detrend array
        np.add.at(y_detr,i,detrend)
        # count number of detrends per sample (depends on overlap)
        np.add.at(counter,i,1)

    # window gaps, marked by missing detrend are set to np.nan
    counter[counter==0] = np.nan
    # create final detrend array
    y_detrend = y_detr/counter       
    if len(y_detrend[np.isnan(y_detrend)]) > 0:
        # replace nan-values assuming a mean of zero
        y_detrend[np.isnan(y_detrend)] = 0.0

    return y_detrend


def exp_detrend(time, y, length, stopper, n_ovrlp):
    x = np.array(time).flatten()
    y = np.array(y).flatten()
    y_detr = np.zeros(shape=(y.shape[0]))
    counter = np.zeros(shape=(y.shape[0]))

    # Define the exponential model
    def exp_model(t, a, b):
        return a * np.exp(-b * t)

    # Calculate the interval and create regular sampled array along time
    interval = length / (n_ovrlp + 1)
    reg_times = np.arange(x[0] - (x[1] - x[0]) - length, x[-1] + length, interval)

    # Extract indices for each interval
    idx = [np.where((x > tt - (length / 2)) & (x <= tt + (length / 2)))[0] for tt in reg_times]

    # Exclude samples without values (np.nan) from exponential detrend
    idx = [i[~np.isnan(y[i])] for i in idx]

    # Only detrend intervals that meet the stopper criteria
    idx = [x for x in idx if len(x) >= stopper]

    for i in idx:
        t = x[i]
        yi = y[i]

        # Initial guess for the parameters
        p0 = [np.max(yi)-np.min(yi), 5]  # a = max(y), b = 0.1

        # Perform non-linear least squares to fit the exponential model
        try:
            popt, pcov = curve_fit(exp_model, t, yi, p0=p0, maxfev=10000)
            a, b = popt
        except RuntimeError as e:
            warnings.warn(f"Error fitting exponential model: {e}")
            a, b = np.nan, np.nan

        # Calculate the detrended values
        model = exp_model(t, a, b)
        detrend = yi - model

        # Add detrended values to detrend array
        np.add.at(y_detr, i, detrend)

        # Count number of detrends per sample (depends on overlap)
        np.add.at(counter, i, 1)

    # Window gaps, marked by missing detrend are set to np.nan
    counter[counter == 0] = np.nan

    # Create final detrend array
    y_detrend = y_detr / counter

    if len(y_detrend[np.isnan(y_detrend)]) > 0:
        # Replace nan-values assuming a mean of zero
        y_detrend[np.isnan(y_detrend)] = 0.0
    print(a,b)
    return y_detrend

#多项式拟合，没有加权，边界效应会稍微强一些
def poly_detrend(time, y, degree=2, length=3, stopper=3, n_ovrlp=2):
    x = np.array(time).flatten()
    y = np.array(y).flatten()
    y_detr = np.zeros(shape=(y.shape[0]))
    counter = np.zeros(shape=(y.shape[0]))

    # Calculate the interval and create regular sampled array along time
    interval = length / (n_ovrlp + 1)
    reg_times = np.arange(x[0] - (x[1] - x[0]) - length, x[-1] + length, interval)

    # Extract indices for each interval
    idx = [np.where((x > tt - (length / 2)) & (x <= tt + (length / 2)))[0] for tt in reg_times]

    # Exclude samples without values (np.nan) from polynomial detrend
    idx = [i[~np.isnan(y[i])] for i in idx]

    # Only detrend intervals that meet the stopper criteria
    idx = [x for x in idx if len(x) >= stopper]

    for i in idx:
        t = x[i]
        yi = y[i]

        # Fit a polynomial to the data in the interval
        p_coeff = np.polyfit(t, yi, degree)  # Fit polynomial of specified degree

        # Calculate the detrended values
        model = np.polyval(p_coeff, t)
        detrend = yi - model

        # Add detrended values to detrend array
        np.add.at(y_detr, i, detrend)

        # Count number of detrends per sample (depends on overlap)
        np.add.at(counter, i, 1)

    # Window gaps, marked by missing detrend are set to np.nan
    counter[counter == 0] = np.nan

    # Create final detrend array
    y_detrend = y_detr / counter

    if len(y_detrend[np.isnan(y_detrend)]) > 0:
        # Replace nan-values assuming a mean of zero
        y_detrend[np.isnan(y_detrend)] = 0.0

    return y_detrend

# 多项式拟合，使用加权最小二乘拟合
def poly_detrend1(time, y, degree, length, stopper, n_ovrlp):
    x = np.array(time).flatten()
    y = np.array(y).flatten()
    y_detr = np.zeros_like(y)
    counter = np.zeros_like(y)

    # 创建规则采样时间点
    interval = length / (n_ovrlp + 1)
    reg_times = np.arange(x[0] - interval, x[-1] + interval, interval)
    idx = [np.where((x > tt - interval/2) & (x <= tt + interval/2))[0] for tt in reg_times]

    # 排除含有 NaN 的样本，并满足 stopper 条件的窗口
    idx = [i[~np.isnan(y[i])] for i in idx]
    idx = [i for i in idx if len(i) >= stopper]

    # 定义多项式模型
    def poly_model(t, *params):
        return np.polyval(params, t)

    # 进行加权最小二乘拟合
    for i in idx:
        t = x[i]
        yi = y[i]
        # 计算权重，边界点权重较低
        weights = 1 / (1 + np.abs((t - x.mean()) / x.mean()))
        weights /= weights.max()  # 归一化权重

        # 获取初始参数估计
        p_initial = np.zeros(degree + 1)

        # 定义每个参数的界限，这里假设参数不会超过 ±100
        #bounds = [(low, high) for low, high in zip(-1000 * np.ones(degree + 1), 1000 * np.ones(degree + 1))]

        try:
            # 执行拟合
            res = least_squares(
                fun=lambda params: np.array(weights * (poly_model(t, *params) - yi)),
                x0=p_initial,
                #bounds=bounds
            )
            p_coeff = res.x
        except Exception as e:
            print(f"Error fitting polynomial: {e}")
            p_coeff = np.nan * t  # 产生一个和 t 形状相同的 NaN 数组

        # 计算去趋势后的值
        model = poly_model(t, *p_coeff)
        detrend = yi - model
        y_detr[i] = detrend

    # 处理 NaN 值
    y_detr[~np.isfinite(y_detr)] = 0

    return y_detr

