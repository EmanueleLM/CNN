# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 18:51:32 2018

@author: Emanuele

Test backpropagation for one layer of 1d-convolution
"""

import convolution as conv
import series_to_matrix as stm

import numpy as np
import sys

sys.path.append('../cnn')


# exponential activation function
def exp(series, weights, stride):
    
    res = conv.conv_1d(series, weights, stride)
    return np.exp(res)

if __name__ == '__main__':
    
    inp = np.random.rand(1, 100)
    
    weights = np.random.rand(1, 10)
    stride = np.random.randint(1,10)
    
    # derivative by implementation (chain rule)
    derivative_by_implementation = exp(inp, weights, stride)@stm.series_to_matrix(inp, weights.shape[1], stride)
    
    # derivative by definition
    derivative_by_def = list()
    epsilon = 1e-5
    for i in range(weights.shape[1]):
        
        weights[:,i] += epsilon
        f_plus = exp(inp, weights, stride)
        weights[:,i] -= 2*epsilon
        f_minus = exp(inp, weights, stride)
        weights[:,i] += epsilon
        
        derivative_by_def.append(np.sum((f_plus-f_minus)/(2*epsilon)))
        
    error = np.linalg.norm(derivative_by_def-derivative_by_implementation)
    print("Error estimating derivative of one layer convolution with implementation is ", error)
 