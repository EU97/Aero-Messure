import numpy as np
from scipy.signal import butter, filtfilt, detrend

def lowpass(x, cutoff_norm=0.3, order=4):
    b,a = butter(order, cutoff_norm, btype='low')
    return filtfilt(b,a,x)

def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.nanmean(x**2))
