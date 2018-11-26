# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:51:36 2018

@author: Emanuele

Dense layer object creator
"""

import numpy as np

class Dense(object):
    
    def __init__(self, shape, activation, output_len):
        
        self.weights = np.zeros(shape=shape)
        self.bias = np.zeros(shape=(1, shape[1]))
        self.act = activation
        # calculate output's length using the input length
        self.output_len = output_len