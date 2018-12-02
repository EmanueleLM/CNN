# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:45:40 2018

@author: Emanuele

Activation functions: provides both the activations functions and the dictionary
 to easy call them.
"""

import numpy as np


def linear(z):
    
    return z;


def exp(z):
    
    return np.exp(z)


def relu(z):
    
    z[z<=0] = 0
    
    return z


dict_activations = { 'linear': linear,
                     'exp': exp,
                     'relu': relu
        }
