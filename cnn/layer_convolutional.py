# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:54:29 2018

@author: Emanuele

Convolutional layer object creator
"""

import activations as act
import convolution as conv
import derivatives as der
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
        self.output = np.zeros(shape=(1, output_len))
        
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
    
    # compute partial derivative of the layer.
    # if the input (input_) is not specified, considers as input the output defined
    #  at self.output
    # Returns the tuple (derivative wrt the output, weights): an 'orchestrator'
    #  will manage to make the derivative of a whole network work, based on the 
    #  output of this function, for each layer.
    def derivative(self, input_=None):
        
        if input_ is not None:
            
            input_ = self.activation(input_)
            
        derivative = der.dict_derivatives[self.act](input_)
        
        return derivative, self.weights
    