# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:40:10 2018

@author: Emanuele

Parameters' initializaton functions: provides both the functions and the dictionary
 to initialize the weights of a given layer.
"""

import numpy as np


def uniform(weights, bias=None, args=None):
    
    if args is None:
        
        if bias is None:
            
            weights = np.random.uniform(.0, 1., size=weights.shape)
            
            return weights
        
        else:
            
            weights = np.random.uniform(.0, 1., size=weights.shape)
            bias = np.random.uniform(.0, 1., size=bias.shape)
            
            return weights, bias
        
    else:
        
        if bias is None:
            
            weights = np.random.uniform(args[0], args[1], size=weights.shape)
            
            return weights
        
        else:
            
            weights = np.random.uniform(args[0], args[1], size=weights.shape)
            bias = np.random.uniform(args[0], args[1], size=bias.shape)
            
            return weights, bias
        

def random(weights, bias=None, args=None):
       
    if args is None:
        
        if bias is None:
            
            weights = np.random.rand(weights.shape[0], weights.shape[1])
            
            return weights
        
        else:
            
            weights = np.random.rand(weights.shape[0], weights.shape[1])
            bias = np.random.rand(bias.shape[0], bias.shape[1])
                        
            return weights, bias
        
    else:
        
        if bias is None:
            
            weights = np.random.rand(weights.shape[0], weights.shape[1])
            
            return weights
        
        else:
            
            weights = np.random.rand(weights.shape[0], weights.shape[1])
            bias = np.random.rand(bias.shape[0], bias.shape[1])
            
            return weights, bias

dict_parameters_init = { 'uniform': uniform,
                         'random': random
                         }
