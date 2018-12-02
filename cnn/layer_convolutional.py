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
        self.input_ = None
        # calculate output's length using the input length
        self.output_len = output_len
        self.output = np.zeros(shape=(1, output_len))
        
        
    # initialize the layer's parameters from a dictionary of methods
    def init_parameters(self, method, parameters=None):
        
        if parameters is None:
            
            self.weights = p_init.dict_parameters_init[method](self.weights)
        
        else:
        
            self.weights = p_init.dict_parameters_init[method](self.weights, None, parameters)
            
        
    """
     Compute the activation function, given the input.
     Takes as input:
         input_:numpy.array, specifies the input for the layer;
         accumulate:boolean: specifies whether the layer stores or not the 
                    input and the activation: they can be both retrieved later
                    to speedup backpropagation phase.
    Returns:
        output:numpy.array, the output of the layer.
    """
    def activation(self, input_, accumulate=False):
        
        if accumulate is True:
            
            self.input_ = input_
        
        output = conv.conv_1d(input_, self.weights, self.stride)
        output = act.dict_activations[self.act](output)
        
        self.output = output
        
        return output
    
    """
     Compute the partial derivative of the layer.
     Takes as input:
         input_:(np.array | None),  specifies the input for the layer. 
                It can be None in case it has been previously stored in the 
                layer's object self.input_.
     Returns:
         (derivative, self.weights):(numpy.array, numpy.array), a tuple 
                                    (derivative wrt the output, weights): an 
                                    'orchestrator' will manage to make the derivative
                                    of a whole network work, based on the
                                    output of this function, for each layer.
    """
    def derivative(self, input_=None):
                   
        derivative = der.dict_derivatives[self.act](input_)
        
        return derivative, self.weights
    