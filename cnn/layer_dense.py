# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:51:36 2018

@author: Emanuele

Dense layer object creator
"""

import activations as act
import derivatives as der
import parameters_init as p_init

import copy as cp
import numpy as np


class Dense(object):
    
    def __init__(self, 
                 shape, 
                 activation, 
                 output_len):
        
        self.weights = np.zeros(shape=shape)
        self.bias = np.zeros(shape=(1, shape[1]))
        self.act = activation
        self.input_ = None
        # set output's length
        self.output_len = output_len
        # vector to store the output of the layer
        self.output = np.zeros(shape=(1, output_len))
        # layer's information
        self.type = 'dense'
        
    
    # initialize the layer's parameters from a dictionary of methods
    def init_parameters(self, method, parameters=None):
        
        if parameters is None:

            self.weights, self.bias = p_init.dict_parameters_init[method](self.weights, self.bias)
        
        else:
        
            self.weights, self.bias = p_init.dict_parameters_init[method](self.weights, self.bias, parameters)
 
    
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
            
        output = np.dot(input_, self.weights) + self.bias
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
    