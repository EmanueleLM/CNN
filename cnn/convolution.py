# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:45:49 2018

@author: Emanuele

Implementation of 1-D convolution
"""

import numpy as np
import sys as sys

"""
Takes as input:
    series:array, shape is (1, n), where n is the number of entries in the series itself;
                 if simd is True, shape is (m, k) where k is the kernel dimension,
                 and m is the number of iterations the kernel is supposed to slide
                 over the series;
    kernel:array, shape is (1, k), where k is the dimension of the kernel itself;
    stride:int, specifies the striding parameter: by default is set to 1.
               the parameter is ignored if simd is set to True;
    simd:boolean, if True, the function is expected to receive a series in matrix
                 form, and iterates on each row which has the same dimension of
                 the kernel. A reduction over the column axis is then applied to
                 obtain the final result.
"""
def conv_1d(series, 
            kernel, 
            striding=1,
            simd=False):
    
    if simd is True:
        
        res = np.multiply(series, kernel)
        res = res.sum(axis=1)
        return res[np.newaxis,:]
    
    # non-SIMD version    
    try:
        res = np.zeros(shape=(1, 
                              int(np.ceil((series.shape[1]-kernel.shape[1]+1)/striding))
                              ))
    except ZeroDivisionError:
        print("\nException: striding cannot be zero\n", file=sys.stderr)
        sys.exit(1)
        
    iter_k = range(0, series.shape[1]-kernel.shape[1]+1, striding)    
    j = 0
    for i in iter_k:
        
        res[:,j] = np.sum(np.multiply(series[:,i:i+kernel.shape[1]], kernel))
        j += 1
    
    return res
        
            