# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:51:41 2018

@author: Emanuele
"""

import convolution as conv
import series_to_matrix as stm

import numpy as np
import sys

sys.path.append('../cnn')

import convolution as conv


if __name__ == '__main__':
    
    np.random.seed(43) # the answer to everything, plus 1
    series = np.random.rand(1, 100)
    
    for k in range(1, 10):
        
        for s in range(1,10):
            
            kernel = np.random.rand(1, k)
            
            # calculate convolution with different couples of (kernels, stridings)
            res = conv.conv_1d(series, kernel, striding=s)
            
            print("Series shape ", series.shape, 
                  "\nKernel shape ", kernel.shape,
                  "\nStride shape ", s,
                  "\nResult shape ", res.shape)
            print("#######\n")
            
            # case where striding is 1 and we have a numpy counterpart function
            if s == 1:
                
                # test convolution, should rise exception if convolution is wrong
                res_numpy = np.correlate(series[0,:], kernel[0,:])
                np.testing.assert_almost_equal(res[0,:], res_numpy)
                
                # test SIMD convolution, should rise exception if convolution is
                #  wrong
                w_series = stm.series_to_matrix(series, kernel.shape[1], s)
                w_res = conv.conv_1d(w_series, kernel, striding=s, simd=True)
                np.testing.assert_almost_equal(w_res[0,:], res_numpy)
            
            # checking differences between simd and non-simd convolution
            w_series = stm.series_to_matrix(series, kernel.shape[1], s)
            w_res = conv.conv_1d(w_series, kernel, striding=s, simd=True)
            np.testing.assert_equal(w_res, res)
        