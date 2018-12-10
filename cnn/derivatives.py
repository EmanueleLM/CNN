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
    
    return z


def tanh(z):
    
    return np.multiply((1-z),(1+z));


def relu(z):
    
    return np.where(z > 0, 1., 0.)


def leaky_relu(z):
        
    return np.where(z > 0, 1., 0.01)


dict_derivatives = { 'linear': linear,
                     'exp': exp,
                     'tanh': tanh,
                     'relu': relu,
                     'leaky_relu': leaky_relu
        }
