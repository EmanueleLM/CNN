# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:19:51 2018

@author: Emanuele

Test backpropagation for a random net.
"""

import copy as cp
import numpy as np
import sys

sys.path.append('../cnn')

import activations as act
import convolution as conv
import nn as nn


# mimic activation of a 2 layer neural net, a convolutional layer followed by a 
#  dense layer
def activation(x, w_conv, stride, act_conv, w_dense, b_dense, act_dense):
    
    # convolution
    tmp = conv.conv_1d(x, w_conv, stride)
    tmp = act.dict_activations[act_conv](tmp)
    # dense layer
    tmp = np.dot(tmp, w_dense) + bias_dense
    tmp = act.dict_activations[act_dense](tmp)
    
    return np.sum(tmp)


if __name__ == '__main__':
    
    net_blocks = {'n_inputs': 100, 
                  'layers': [
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 25), 'stride': 5}, 
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 32)}
                          ]
                      }
    
    net = nn.NN(net_blocks)
    
    # initialize the parameters
    net.init_parameters(['uniform', -1., 1.]) 

    # calculate partial derivative for each layer
    input_ = np.random.rand(1, net.n_inputs)
    # activate the net for a random input
    net.activation(input_, accumulate=True)    
    # calculate partial derivative for each layer
    net.derivative(None)
    target = cp.copy(net.output)
    # use as loss the 'none' function which returns the prediction itself
    net.backpropagation(target, 'none', 'ada', 1e-3)
        
    derivative_by_calc = np.concatenate([net.layers[0].delta_weights.flatten(),
                               net.layers[1].delta_weights.flatten(),
                               net.layers[1].delta_bias.flatten()
                               ]).flatten()
    
    # calculate derivative with the definition
    weights_conv = cp.copy(net.layers[0].weights)
    stride_conv = net.layers[0].stride
    act_conv = net.layers[0].act
    weights_dense = cp.copy(net.layers[1].weights)
    bias_dense = cp.copy(net.layers[1].bias)
    act_dense = net.layers[1].act
    
    derivative_by_def = list()
    epsilon = 1e-5
    
    # convolutional layer: weights
    for i in range(weights_conv.shape[1]):
        
        weights_conv[:,i] += 2*epsilon
        
        f_plus = activation(input_,
                            weights_conv, stride_conv, act_conv, 
                            weights_dense, bias_dense, act_dense)
        
        weights_conv[:,i] -= 2*epsilon

        f_minus = activation(input_,
                            weights_conv, stride_conv, act_conv, 
                            weights_dense, bias_dense, act_dense)
        
        weights_conv[:,i] += epsilon
                            
        derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
    
    # dense layer: weights
    for i in range(weights_dense.shape[0]):
        
        for j in range(weights_dense.shape[1]):
            
            weights_dense[i,j] += 2*epsilon
        
            f_plus = activation(input_,
                                weights_conv, stride_conv, act_conv, 
                                weights_dense, bias_dense, act_dense)
            
            weights_dense[i,j] -= 2*epsilon
            
            f_minus = activation(input_,
                                weights_conv, stride_conv, act_conv, 
                                weights_dense, bias_dense, act_dense)
    
            weights_dense[i,j] += epsilon
                                            
            derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
    
    # dense layer: bias
    for i in range(bias_dense.shape[1]):
                 
        bias_dense[:,i] += 2*epsilon
    
        f_plus = activation(input_,
                            weights_conv, stride_conv, act_conv, 
                            weights_dense, bias_dense, act_dense)
        
        bias_dense[:,i]-= 2*epsilon
        
        f_minus = activation(input_,
                            weights_conv, stride_conv, act_conv, 
                            weights_dense, bias_dense, act_dense)

        bias_dense[:,i] += epsilon
                            
        derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
        
    # evaluate error
    derivative_by_def = np.array(derivative_by_def).flatten()
    
    error = np.linalg.norm(derivative_by_calc-derivative_by_def)
    
    print("Error: ", error)        
        
