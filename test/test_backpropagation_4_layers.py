# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:19:51 2018

@author: Emanuele

Test backpropagation for a random 4 layers net: 2 ccn layers and 2 dense layers.
"""

import copy as cp
import numpy as np
import sys

sys.path.append('../cnn')

import activations as act
import convolution as conv
import nn as nn


# mimic activation of a 4 layer neural net, two convolutional layers followed 
#  by two dense layers
def activation(x, w_conv_1, stride_1, act_conv_1, w_conv_2, stride_2, act_conv_2, 
               w_dense_1, b_dense_1, act_dense_1, w_dense_2, b_dense_2, act_dense_2):
    
    # convolutions
    tmp = conv.conv_1d(x, w_conv_1, stride_1)
    tmp = act.dict_activations[act_conv_1](tmp)
    
    tmp = conv.conv_1d(tmp, w_conv_2, stride_2)
    tmp = act.dict_activations[act_conv_2](tmp)    
    
    # dense layers
    tmp = np.dot(tmp, w_dense_1) + b_dense_1
    tmp = act.dict_activations[act_dense_1](tmp)
    
    tmp = np.dot(tmp, w_dense_2) + b_dense_2
    tmp = act.dict_activations[act_dense_2](tmp)
    
    return np.sum(tmp)


if __name__ == '__main__':
    
    net_blocks = {'n_inputs': 50, 
                  'layers': [
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 5), 'stride': 2}, 
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 10), 'stride': 2},
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 50)},
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 10)}
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
                               net.layers[2].delta_weights.flatten(),
                               net.layers[2].delta_bias.flatten(),
                               net.layers[3].delta_weights.flatten(),
                               net.layers[3].delta_bias.flatten(),
                               ]).flatten()
    
    # initialize the weights to the same values as the net
    weights_conv_1 = cp.copy(net.layers[0].weights)
    stride_conv_1 = net.layers[0].stride
    act_conv_1 = net.layers[0].act
    weights_conv_2 = cp.copy(net.layers[1].weights)
    stride_conv_2 = net.layers[1].stride
    act_conv_2 = net.layers[1].act
    
    weights_dense_1 = cp.copy(net.layers[2].weights)
    bias_dense_1 = cp.copy(net.layers[2].bias)
    act_dense_1 = net.layers[2].act
    weights_dense_2 = cp.copy(net.layers[3].weights)
    bias_dense_2 = cp.copy(net.layers[3].bias)
    act_dense_2 = net.layers[3].act
    
    # calculate derivative with the definition
    derivative_by_def = list()
    epsilon = 1e-5
    
    # first convolutional layer: weights
    for i in range(weights_conv_1.shape[1]):
        
        weights_conv_1[:,i] += 2*epsilon
        
        f_plus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
        
        weights_conv_1[:,i] -= 2*epsilon

        f_minus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
        
        weights_conv_1[:,i] += epsilon
                            
        derivative_by_def.append((f_plus-f_minus)/(2*epsilon))

    # second convolutional layer: weights
    for i in range(weights_conv_2.shape[1]):
        
        weights_conv_2[:,i] += 2*epsilon
        
        f_plus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
        
        weights_conv_2[:,i] -= 2*epsilon

        f_minus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
        
        weights_conv_2[:,i] += epsilon
                            
        derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
    
    # first dense layer: weights
    for i in range(weights_dense_1.shape[0]):
        
        for j in range(weights_dense_1.shape[1]):
            
            weights_dense_1[i,j] += 2*epsilon
        
            f_plus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
            
            weights_dense_1[i,j] -= 2*epsilon
            
            f_minus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
    
            weights_dense_1[i,j] += epsilon
                                            
            derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
    
    # first dense layer: bias
    for i in range(bias_dense_1.shape[1]):
                 
        bias_dense_1[:,i] += 2*epsilon
    
        f_plus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
        
        bias_dense_1[:,i]-= 2*epsilon
        
        f_minus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)

        bias_dense_1[:,i] += epsilon
                            
        derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
        
    # second dense layer: weights
    for i in range(weights_dense_2.shape[0]):
        
        for j in range(weights_dense_2.shape[1]):
            
            weights_dense_2[i,j] += 2*epsilon
        
            f_plus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
            
            weights_dense_2[i,j] -= 2*epsilon
            
            f_minus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
    
            weights_dense_2[i,j] += epsilon
                                            
            derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
    
    # second dense layer: bias
    for i in range(bias_dense_2.shape[1]):
                 
        bias_dense_2[:,i] += 2*epsilon
    
        f_plus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)
        
        bias_dense_2[:,i]-= 2*epsilon
        
        f_minus = activation(input_,
                            weights_conv_1, stride_conv_1, act_conv_1, 
                            weights_conv_2, stride_conv_2, act_conv_2,
                            weights_dense_1, bias_dense_1, act_dense_1,
                            weights_dense_2, bias_dense_2, act_dense_2)

        bias_dense_2[:,i] += epsilon
                            
        derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
        
    # evaluate error
    derivative_by_def = np.array(derivative_by_def).flatten()
    
    error = np.linalg.norm(derivative_by_calc-derivative_by_def)
    
    print("Error: ", error)        
        
