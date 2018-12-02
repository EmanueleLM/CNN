# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:46:57 2018

@author: Emanuele

Test 1-dimensional pooling
"""

import pooling as pool
import series_to_matrix as stm

import numpy as np
import sys

sys.path.append('../cnn')


if __name__ == '__main__':
    
    np.random.seed(41) # the answer to everything, minus 1
    series = np.random.rand(1, 100)
    
    # shape-based test
    for skip_factor in range(1, 10):
        
        method = ('max' if np.random.randint(0,2) == 0 else 'average')  
        res = pool.pooling_1d(series, stride=skip_factor, method=method)
        
        print("Series shape ", series.shape, 
              "\nSkip parameter shape ", skip_factor,
              "\nMethod ", method,
              "\nResult shape ", res.shape)
        print("#######\n")
        
    # result-based controlled test
    series_int = np.array([i for i in range(100)])[np.newaxis,:]
    
    for skip_factor in range(1, 10):
    
        method = ('max' if np.random.randint(0,2) == 0 else 'average')  
        res = pool.pooling_1d(series_int, stride=skip_factor, method=method)
        
        print("Result ", res, 
              "\nSkip parameter shape ", skip_factor,
              "\nMethod ", method)
        print("#######\n")
    
    # SIMD test
    for skip_factor in range(1,10):
        
        method = ('max' if np.random.randint(0,2) == 0 else 'average') 
        w_res = stm.series_to_matrix(series_int, 1, skip_factor)
        w_res = pool.pooling_1d(w_res, stride=1, method=method, simd=True)
        
        print("Series shape ", w_res.shape, 
              "\nSkip parameter shape ", skip_factor,
              "\nMethod ", method,
              "\nResult shape ", w_res.shape)
        print("#######\n")
    
    # measure distance between simd and non-simd versions, picking random values
    #  for both the skip and the series' values
    skip = np.random.randint(1,100)
    s = np.random.rand(1, 1000)    
    w_s = stm.series_to_matrix(s, skip, skip)
    
    pool_non_simd = pool.pooling_1d(series=s, stride=skip, method='max', simd=False)
    pool_simd = pool.pooling_1d(series=w_s, stride=None, method='max', simd=True)
    
    if np.linalg.norm(pool_non_simd-pool_simd) != 0:
        
        print("Error, pooling and pooling SIMD don't match.", file=sys.stderr)
    
