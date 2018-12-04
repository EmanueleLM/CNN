# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:55:42 2018

@author: Emanuele

Test activation's output of a neural network.
"""

import copy as cp
import numpy as np
import sys as sys

sys.path.append('../cnn')

import activations as act
import convolution as conv
import layer_convolutional as layer_conv
import layer_dense as layer_dense
import nn as nn


if __name__ == '__main__':
    
    net_blocks = {'n_inputs': 1000, 
                  'layers': [
                          {'type': 'conv', 'activation': 'exp', 'shape': (1, 5), 'stride': 5}, 
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 4), 'stride': 2},
                          {'type': 'conv', 'activation': 'exp', 'shape': (1, 7), 'stride': 3},
                          {'type': 'dense', 'activation': 'linear', 'shape': (None, 150)},
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 20)}
                          ]
                  }
    
    # create the net
    net = nn.NN(net_blocks)
    
    # initialize the parameters to random values between [-1, 1]
    net.init_parameters(['uniform', -1., 1.])
    
    input_ = np.random.rand(1, net.n_inputs)
    
    tmp = cp.copy(input_)
    for i in range(len(net.layers)):
        
        if isinstance(net.layers[i], layer_conv.Conv):
            
            weights = net.layers[i].weights
            stride = net.layers[i].stride
            tmp = conv.conv_1d(tmp, weights, stride)
            tmp = act.dict_activations[net.layers[i].act](tmp)
        
        elif isinstance(net.layers[i], layer_dense.Dense): 
            
            weights = net.layers[i].weights
            bias = net.layers[i].bias
            tmp = np.dot(tmp, weights) + bias
            tmp = act.dict_activations[net.layers[i].act](tmp)
            
        output_by_definition = tmp

    # calculate net's output 
    net.activation(input_)
    output_by_calculation = cp.copy(net.output)
    
    error = np.linalg.norm(output_by_calculation- output_by_definition)
    print("Error = ", error)
        
 