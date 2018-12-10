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


def tanh(z):
    
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def relu(z):
    
    return np.where(z > 0, z, 0.)


def leaky_relu(z):
        
    return np.where(z > 0, z, z * 0.01)


dict_activations = { 'linear': linear,
                     'exp': exp,
                     'tanh': tanh,
                     'relu': relu,
                     'leaky_relu': leaky_relu
        }
