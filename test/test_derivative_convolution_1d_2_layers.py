# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:42:39 2018

@author: Emanuele

Test backpropagation for two layers of 1d-convolution, linear activation
"""

import convolution as conv
import series_to_matrix as stm

import numpy as np
import sys

sys.path.append('../cnn')


# linear activation function
def linear(series, weights, stride):
    
    res = conv.conv_1d(series, weights, stride)
    return res

# exponential activation function
def exp(series, weights, stride):
    
    res = conv.conv_1d(series, weights, stride)
    return np.exp(res)

if __name__ == '__main__':
    
    inp = np.random.rand(1, 1000)
    
    weights = np.random.rand(1, np.random.randint(1, 10))
    stride = np.random.randint(1, 10)
    
    weights_c = np.random.rand(1, np.random.randint(1, 10))
    stride_c = np.random.randint(1, 10)
    
    ### linear activation test ###
    # derivative by implementation (chain rule)
    derivative_by_implementation =  stm.series_to_matrix(inp, weights.shape[1], stride).T
    tmp = np.zeros(shape=(weights.shape[1], linear(linear(inp, weights, stride), weights_c, stride_c).shape[1]))
    for i in range(derivative_by_implementation.shape[0]):
        tmp[i] = conv.conv_1d(derivative_by_implementation[np.newaxis,i], weights_c, stride_c)
    derivative_by_implementation = np.sum(tmp, axis=1)
    
    # derivative by definition
    derivative_by_def = list()
    epsilon = 1e-5
    for i in range(weights.shape[1]):
        
        weights[:,i] += epsilon
        f_plus = linear(linear(inp, weights, stride), weights_c, stride_c)
        weights[:,i] -= 2*epsilon
        f_minus = linear(linear(inp, weights, stride), weights_c, stride_c)
        weights[:,i] += epsilon
        
        derivative_by_def.append(np.sum((f_plus-f_minus)/(2*epsilon)))
        
    error = np.linalg.norm(derivative_by_def-derivative_by_implementation)
    print("Error estimating derivative of two layers linear convolution with implementation is ", error)
    
    ### exponential + linear activation test ###
    # derivative by implementation (chain rule)
    derivative_by_implementation =  exp(inp, weights, stride)*stm.series_to_matrix(inp, weights.shape[1], stride).T
    tmp = np.zeros(shape=(weights.shape[1], exp(exp(inp, weights, stride), weights_c, stride_c).shape[1]))
    for i in range(derivative_by_implementation.shape[0]):
        tmp[i] = conv.conv_1d(derivative_by_implementation[np.newaxis,i], weights_c, stride_c)
    derivative_by_implementation = np.sum(tmp, axis=1)
    
    # derivative by definition
    derivative_by_def = list()
    epsilon = 1e-5
    for i in range(weights.shape[1]):
        
        weights[:,i] += epsilon
        f_plus = linear(exp(inp, weights, stride), weights_c, stride_c)
        weights[:,i] -= 2*epsilon
        f_minus = linear(exp(inp, weights, stride), weights_c, stride_c)
        weights[:,i] += epsilon
        
        derivative_by_def.append(np.sum((f_plus-f_minus)/(2*epsilon)))
        
    error = np.linalg.norm(derivative_by_def-derivative_by_implementation)
    print("Error estimating derivative of two layers (linear+exponential) convolution with implementation is ", error)
