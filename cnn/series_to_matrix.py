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
    series:numpy.array, a (1,n) array;
    k_shape:int, specifies the kernel shape;
    stride:int, specifies striding value, i.e. how many values are skipped at a
               new iteration.
Returns:
    res:matrix, a (i, k_shape) matrix that can be used to cast convolution as SIMD multiply
               operation: i can be calculated as $\ceil{\dfrac{len(series)-k_shape+1}{striding}}$
"""
def series_to_matrix(series, k_shape, striding=1):
    
    res = np.zeros(shape=(int((series.shape[1]-k_shape)/striding)+1,
                   k_shape))
    
    j = 0
    for i in range(0, series.shape[1]-k_shape+1, striding):
        
        res[j] = series[:,i:i+k_shape]
        j += 1
    
    return res


"""
Takes as input:
    matrix:numpy.array, a (n,m) array;
    k_shape:int, specifies the kernel shape;
    stride:int, specifies striding value, i.e. how many values are skipped at a
               new iteration.
Returns:
    res:matrix, a (i, j, k_shape) tensor, where i is the number of kernels, 
        j is the length of each row after series_to_matrix is applied
"""
def matrix_to_tensor(matrix, k_shape, striding=1):
    
    res = np.zeros(shape=(matrix.shape[1], 
                          int((matrix.shape[0]-k_shape)/striding)+1,
                          k_shape))
    
    for i in range(len(res)):
        
        res[i] = series_to_matrix(matrix[np.newaxis,:,i], k_shape, striding)
        
    return res


"""
Takes as input:
    tensor:numpy.array, a (k_n, k_w, m) array;
    k_shapes:tuple, specifies the kernel shapes kp_n, kp_w;
    stride:int, specifies striding value, i.e. how many values are skipped at a
               new iteration.
Returns:
    res:tensor, a (k_n, k_w, out, kp_n, kp_w) 5 dimensional tensor, where k_n, k_w
         are the dimensions of the initial kernel (the ones to be preserved in 
         derivative phase), out is the output dimensions after the convolution,
         kp_n, kp_w are the dimensions of the kernel used in the convolution.
"""
def tensor_to_tensor(tensor, k_shapes, striding=1):
    
    res = np.zeros(shape=(tensor.shape[0], tensor.shape[1],
                          int((tensor.shape[2]-k_shapes[1])/striding)+1,
                          k_shapes[0], k_shapes[1]))
    
    for i in range(tensor.shape[0]):
        
        for j in range(tensor.shape[1]):
            
            for l in range(k_shapes[0]):
                      
                res[i,j,:,l] = series_to_matrix(tensor[np.newaxis,i,j], k_shapes[1], striding)
        
    return res


"""
Takes as input:
    tensor:numpy.array, a (1, m) array;
    k_shapes:tuple, specifies the kernel shapes k_n, k_w;
    stride:int, specifies striding value, i.e. how many values are skipped at a
               new iteration.
Returns:
    res:tensor, a (k_n, k_w, out) 3 dimensional tensor, where k_n, k_w
         are the dimensions of the initial kernel (the ones to be preserved in 
         derivative phase), out is the output dimensions after the convolution.
"""
def series_to_tensor(tensor, k_shapes, striding=1):

    res = np.zeros(shape=(k_shapes[0], k_shapes[1],
                          int((tensor.shape[1]-k_shapes[1])/striding)+1))
        
    for i in range(tensor.shape[0]):
                      
            res[i,:] = series_to_matrix(tensor[np.newaxis,i], k_shapes[1], striding).T
        
    return res    
    
                          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    