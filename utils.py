import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
# Filter requirements.

cutoff = 0.8  
SOS = b,a = butter(
    N = 4, 
    Wn = cutoff, 
    btype='low', 
    analog=False, 
    output='sos')

def noise_filter(y):
    n = 2  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    return lfilter(b, a, y)

def data_preprocessing(final):
    for col in ["AccelerometerX","AccelerometerY","AccelerometerZ"]:
            final[col] = sosfilt( SOS, final[col].values)
    for col in ["GyroscopeX",	"GyroscopeY",	"GyroscopeZ"]:
        final[col] = noise_filter( final[col].values )
    final['GyroscopeAbsolute'] = np.sqrt(final[["GyroscopeX",	"GyroscopeY",	"GyroscopeZ"]].sum(axis=1)**2)
    final['AccelerometerAbsolute'] = np.sqrt(final[["AccelerometerX",	"AccelerometerY",	"AccelerometerZ"]].sum(axis=1)**2)

    return final