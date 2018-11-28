# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:54:29 2018

@author: Emanuele

Convolutional layer object creator
"""

import activations as act
import convolution as conv
import parameters_init as p_init

import numpy as np

class Conv(object):
    
    def __init__(self, 
                 shape, 
                 activation, 
                 stride, 
                 output_len):
         
        self.weights = np.zeros(shape=shape)
        self.act = activation
        self.stride = stride
        # calculate output's length using the input length
        self.output_len = output_len
        
    # initialize the layer's parameters from a dictionary of methods
    def init_parameters(self, method, parameters=None):
        
        if parameters is None:
            
            self.weights = p_init.dict_parameters_init[method](self.weights)
        
        else:
        
            self.weights = p_init.dict_parameters_init[method](self.weights, None, parameters)
        
    # activation function
    def activation(self, input_):
        
        output = conv.conv_1d(input_, self.weights, self.stride)
        output = act.dict_activations[self.act](output)
        
        return output
    