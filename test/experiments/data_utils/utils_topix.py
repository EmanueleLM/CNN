# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:57:21 2018

@author: Emanuele

Data utilities for TOPIX prediction
"""

import numpy as np
import pandas as pd
import scipy.stats as st


"""
 Measure randomness of a binary string.
 Takes as input:
     input_:numpy.array is the binary vector;
     significance:float, the significance of the test.
"""
def random_test(input_, significance=5e-2):
    
    input_len = len(input_)
    r = input_len
    
    ones = np.sum(input_)
    zeros = input_len - ones
    
    r_hat = 2*(ones*zeros)/(ones+zeros) + 1 
    s_r = ((2*ones*zeros)*(2*ones*zeros-ones-zeros))/((zeros+ones-1)*(ones+zeros)**2)
    
    z = (r - r_hat)/s_r
    print(r, r_hat, s_r, st.norm.ppf(1-significance/2))
    
    # test is not random with this significance
    if np.abs(z) > st.norm.ppf(1-significance/2):
        
        return False  
    
    else:
        
        return True
    

"""
 Measure autocorrelation of a sequence with the lag test from Box and Jenkins, 1976.
 Takes as input:
     input_:numpy.array is the binary vector;
     lag:int, the time-lag used to measure autocorrelation of the input sequence.
     tolerance:float, the tolerance of the test.
"""
def autocorrelation_test(input_, lag, tolerance=1e-2):
    
    input_len = len(input_)
    mean = input_.mean()
    r_k_num = r_k_den = 0.
    
    for i in range(input_len-lag-1):
        
        r_k_num += (input_[i]-mean)*(input_[i+lag]-mean)
        r_k_den += (input_[i]-mean)**2
    
    if np.abs(r_k_num/(r_k_den+1e-10)) <= tolerance/2:  # two tail test
        
        return True
    
    else:
        
        return False    
    

"""
 Gaussian pdf estimator.
 Takes as input:
     x:numpy.array, the input vector;
     mean:float, the mean of the gaussian distribution;
     variance:float, the variance of the gaussian distribution.
"""
def gaussian_pdf(x, mean, variance):

    p_x = (1 / (2 * np.pi * variance) ** .5) * np.exp(-((x - mean) ** 2) / (2 * variance))

    return p_x


"""
 Turn a series into a matrix (i.e. repeated batches).
"""
def series_to_matrix(series, k_shape, striding=1):
    
    res = np.zeros(shape=(int((series.shape[0] - k_shape) / striding) + 1,
                          k_shape)
                   )
    j = 0
    for i in range(0, series.shape[0] - k_shape + 1, striding):
        res[j] = series[i:i + k_shape]
        j += 1

    return res


"""
 3 modes are possible and need do be specified in 'mode' variable:
    'train': all data is reserved to train;
    'train-test': split between train and test, according to non_train_percentage;
    'validation': data is split among train, test and validation: their percentage is chosen according to the percantge
                  of data that has not been included in train (1-non_train_percentage) and assigned to validation
                  proportionally to val_rel_percentage.
"""
def generate_batches(filename, 
                     window, 
                     mode='train-test', 
                     non_train_percentage=.7, 
                     val_rel_percentage=.5):
    
    data = pd.read_csv(filename, delimiter=',', header=0)
    data = (data.iloc[:, 0]).values

    if mode == 'train':

        y = data[window:]
        x = series_to_matrix(data, window, 1)[:-1]

        return x, y

    elif mode == 'train-test':

        train_size = int((1 - non_train_percentage) * np.ceil(len(data)))
        y_train = data[window:train_size]
        x_train = series_to_matrix(data, window, 1)[:train_size - window]
        y_test = data[train_size:]
        x_test = series_to_matrix(data, window, 1)[train_size:]

        return x_train, y_train, x_test, y_test

    elif mode == 'validation':

        # split between train and validation+test
        train_size = int((1 - non_train_percentage) * np.ceil(len(data)))
        y_train = data[window:train_size]
        x_train = series_to_matrix(data, window, 1)[:train_size - window]

        # split validation+test into validation and test
        validation_size = int(val_rel_percentage * np.ceil(len(data) * non_train_percentage))
        y_val = data[train_size:train_size + validation_size]
        x_val = series_to_matrix(data, window, 1)[train_size - window:train_size + validation_size - window]
        y_test = data[train_size + validation_size:]
        x_test = series_to_matrix(data, window, 1)[train_size + validation_size - window:-window]

        return x_train, y_train, x_val, y_val, x_test, y_test
    