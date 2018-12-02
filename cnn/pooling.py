# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:08:27 2018

@author: Emanuele

Implementation of pooling strategies
"""

import numpy as np


"""
Takes as input:
    series:array, shape is (1, n), where n is the number of entries in the series itself;
                 if simd is True, shape is (m, s) where s is the size of the pooling 'mask',
                 and m is the number of iterations the pooling 'mask' is supposed to slide
                 over the series;
    stride:int, specifies the pooling 'mask' size: by default is set to 1.
               the parameter is ignored if simd is set to True;
    method:string, specifies the method used for pooling, by default is 'max';
    simd:boolean, if True, the function is expected to receive a series in matrix
                 form, and iterates on each row which has the same dimension of
                 the pooling 'mask'. A reduction over the column axis is then applied to
                 obtain the final result.
"""
def pooling_1d(series, 
               stride=1, 
               method='max', 
               simd=False):
    
    # impose pooling method
    if method == 'max':
        
        pool_method = np.max
        
    elif method == 'average':
        
        pool_method = np.average
        
    else:
        
        pool_method = np.max
    
    if simd is True:

        res = pool_method(series, axis=1)            
        return res[np.newaxis,:]
    
    # ignore pooling
    if stride==0:
        
        return series;
    
    # non-SIMD version           
    res = np.zeros(shape=(1, int(np.ceil(series.shape[1]/stride))))
    
    iter_k = range(0, series.shape[1], stride)
    j = 0
    for i in iter_k:
        
        print(series[:,i:i+stride])
        res[:,j] = pool_method(series[:,i:i+stride])
        j += 1

    return res
    