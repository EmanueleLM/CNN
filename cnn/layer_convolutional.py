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

import copy as cp
import numpy as np


class Conv(object):
    
    def __init__(self, 
                 shape, 
                 activation, 
                 stride, 
                 output_len):
        
        # layer's information
        self.type = 'conv'
        
        # layer's parameters
        self.weights = np.zeros(shape=shape)
        self.act = activation
        self.stride = stride
        self.input_ = None
        
        # calculate output's length using the input length
        self.output_len = output_len
        self.output = np.zeros(shape=(1, output_len))
        
        # storage for the parameter's update
        self.delta_weights = np.zeros(shape=shape)
        
        
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
            
            self.input_ = cp.copy(input_)
        
        output = conv.conv_1d(input_, self.weights, self.stride)
        output = act.dict_activations[self.act](output)
        
        self.output = cp.copy(output)
        
        return output
    
    
    """
     Compute the partial derivative of the layer.
     Takes as input:
         input_:(np.array | None),  specifies the input for the layer. 
                It can be None in case it has been previously stored in the 
                layer's object self.input_. Please note that if the input is None
                the output of the layer needs to be calculated in place, but the 
                result is not stored (as it happens in activation function when
                accumulate is set to True).
     Returns:
         (derivative, self.weights):(numpy.array, numpy.array), a tuple 
                                    (derivative wrt the output, weights): an 
                                    'orchestrator' will manage to make the derivative
                                    of a whole network work, based on the
                                    output of this function, for each layer.
    """
    def derivative(self, input_=None):
                   
        if input_ is not None:
            
            output = self.activation(input_)
            derivative = der.dict_derivatives[self.act](output)
            
        else:

            derivative = der.dict_derivatives[self.act](self.output)
        
        return derivative, self.weights
   
    
    """
     Update the parameters of the layer: til now only classic sgd is available
    """
    def parameters_update(self, optimizer, l_rate):
        
        self.weights -= l_rate*self.delta_weights
    