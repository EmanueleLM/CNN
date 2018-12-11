# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:57:21 2018

@author: Emanuele

Data utilities for TOPIX prediction
"""

import numpy as np
import pandas as pd

"""
 Gaussian pdf estimator.
"""
def gaussian_pdf(x, mean, variance):

    p_x = (1 / (2 * np.pi * variance) ** .5) * np.exp(-((x - mean) ** 2) / (2 * variance))

    return p_x


"""
 Turn a series into a matrix (i.e. repeated batches).
"""
def series_to_matrix(series,
                     k_shape,
                     striding=1):
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
def generate_batches(filename, window, mode='train-test', non_train_percentage=.7, val_rel_percentage=.5):
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
    