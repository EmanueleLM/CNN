# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:51:36 2018

@author: Emanuele

Dense layer object creator
"""

import activations as act
import parameters_init as p_init

import numpy as np

class Dense(object):
    
    def __init__(self, 
                 shape, 
                 activation, 
                 output_len):
        
        self.weights = np.zeros(shape=shape)
        self.bias = np.zeros(shape=(1, shape[1]))
        self.act = activation
        # set output's length
        self.output_len = output_len
    
    # initialize the layer's parameters from a dictionary of methods
    def init_parameters(self, method, parameters=None):
        
        if parameters is None:

            self.weights, self.bias = p_init.dict_parameters_init[method](self.weights, self.bias)
        
        else:
        
            self.weights, self.bias = p_init.dict_parameters_init[method](self.weights, self.bias, parameters)
                    
    
    # activation function
    def activation(self, input_):
        
        output = np.dot(input_, self.weights) + self.bias
        output = act.dict_activations[self.act](output)
        
        return output