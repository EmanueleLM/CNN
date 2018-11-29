# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:13:56 2018

@author: Emanuele

Derivative of each activation function: provides both the activations functions
 and the dictionary to easy call them.
"""

import numpy as np

def linear(z):
    
    return np.ones(shape=z.shape);

def exp(z):
    
    return np.exp(z)

def relu(z):
    
    z[z<=0] = 0.
    z[z>0] = 1.
    
    return z

dict_derivatives = { 'linear': linear,
                     'exp': exp,
                     'relu': relu
        }