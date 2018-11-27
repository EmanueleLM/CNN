# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:54:29 2018

@author: Emanuele

Convolutional layer object creator
"""

import activation_dictionaries as ad
import convolution as conv

import numpy as np

class Conv(object):
    
    def __init__(self, shape, activation, stride, output_len):
         
        self.weights = np.zeros(shape=shape)
        self.act = activation
        self.stride = stride
        # calculate output's length using the input length
        self.output_len = output_len
        
    # activation function
    def activation(self, input_):
        
        output = conv.conv_1d(input_, self.weights, self.stride)
        output = ad.dict_activations[self.act](output)
        
        return output
    