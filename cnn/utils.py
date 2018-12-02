# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:39:01 2018

@author: Emanuele

Utils functions like convolutional shape calculator
"""


def conv_shape(input_shape, kernel_shape, stride):
    
    return int((input_shape-kernel_shape)/stride)+1