# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:05:40 2018

@author: Emanuele

Transform a series into a matrix, based on striding and kernel's shape value.
This way SIMD operations can be used with convolution: in particular from 
 convolution one may use conv_1d with simd flag set to True.
This function is employed, among the others, in the backpropagation phase to 
 calculate the contribution of an activation function to the variables that 
 compose it.
"""

import numpy as np

"""
Takes as input:
    series:array, a (1,n) array;
    k_shape:int, specifies the kernel shape;
    stride:int, specifies striding value, i.e. how many values are skipped at a
               new iteration.
Returns:
    res:matrix, a (i, k_shape) matrix that can be used to cast convolution as SIMD multiply
               operation: i can be calculated as $\ceil{\dfrac{len(series)-k_shape+1}{striding}}$
"""
def series_to_matrix(series, 
                     k_shape, 
                     striding=1):
    
    res = np.zeros(shape=(int((series.shape[1]-k_shape)/striding)+1,
                   k_shape)
                      )
    j = 0
    for i in range(0, series.shape[1]-k_shape+1, striding):
        
        res[j] = series[:,i:i+k_shape]
        j += 1
    
    return res
    