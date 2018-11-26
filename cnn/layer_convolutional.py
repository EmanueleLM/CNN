# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:54:29 2018

@author: Emanuele

Convolutional layer object creator
"""

import utils as utils

import numpy as np

class Conv(object):
    
    def __init__(self, shape, activation, stride, output_len):
         
        self.weights = np.zeros(shape=shape)
        self.bias = np.zeros(shape=(1, shape[1]))
        self.act = activation
        self.stride = stride
        # calculate output's length using the input length
        self.output_len = output_len