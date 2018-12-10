# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:19:51 2018

@author: Emanuele

Test backpropagation for a random layers net.
"""

import copy as cp
import numpy as np
import sys

sys.path.append('../cnn')

import activations as act
import convolution as conv
import nn as nn


# mimic activation of a layer neural net
def activation(input_, 
               w_conv, stride, act_conv, 
               w_dense, b_dense, act_dense):
    
    tmp = input_
    
    # convolutions
    for i in range(len(w_conv)):
        
        tmp = conv.conv_1d(tmp, w_conv[i], stride[i])
        tmp = act.dict_activations[act_conv[i]](tmp)  
    
    # dense layers
    for i in range(len(w_dense)):
        
        tmp = np.dot(tmp, w_dense[i]) + b_dense[i]
        tmp = act.dict_activations[act_dense[i]](tmp)
    
    return tmp


if __name__ == '__main__':
    
    net_blocks = {'n_inputs': 24, 
                  'layers': [
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 3), 'stride': 2}, 
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 4), 'stride': 1},
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 3), 'stride': 1},
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 35)},                    
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 1)}
                          ]
                  }
    
    net = nn.NN(net_blocks)
    
    # initialize the parameters
    net.init_parameters(['uniform', -.5, 1.])

    # calculate partial derivative for each layer
    input_ = np.random.rand(1, net.n_inputs)
    
    # activate the net for a random input
    net.activation(input_, accumulate=True)
    
    # calculate partial derivative for each layer
    net.derivative(None)
    
    # define the target as the net's output
    target = cp.copy(net.output)
    
    # use as loss the 'none' function which returns the prediction itself
    net.backpropagation(target, 'none', 'ada', 1e-3)
    
    
    # append the updates calculated by the net
    derivative_by_calc = np.array([])
    
    for l in net.layers:
        
        if l.type == 'dense':
            
            derivative_by_calc = np.append(derivative_by_calc, 
                                           l.delta_weights.flatten()
                                           )
            derivative_by_calc = np.append(derivative_by_calc, 
                                   l.delta_bias.flatten()
                                   )
        
        elif l.type == 'conv':
            
            derivative_by_calc = np.append(derivative_by_calc, 
                               l.delta_weights.flatten()
                               )
    
    derivative_by_calc = derivative_by_calc.flatten()
    
    # calculate derivative with the definition
    weights_conv = list([])
    stride_conv = list([])
    act_conv = list([])
    weights_dense = list([])
    bias_dense = list([])
    act_dense = list([])
    
    for l in net.layers:
        
        if l.type == 'dense':
            
            weights_dense.append(l.weights)
            bias_dense.append(l.bias)
            act_dense.append(l.act)
            
        elif l.type == 'conv':
            
            weights_conv.append(l.weights)
            stride_conv.append(l.stride)
            act_conv.append(l.act)
    
    derivative_by_def = list()
    epsilon = 1e-5
    n_dense = n_conv = 0
    
    for n in range(len(net.layers)):
                               
        # dense layer
        if net.layers[n].type == 'dense':
                          
            # weights
            for i in range(net.layers[n].weights.shape[0]):
                
                for j in range(net.layers[n].weights.shape[1]):
                
                    weights_dense[n_dense][i,j] += 2*epsilon
                    
                    f_plus = np.sum(activation(input_,
                                        weights_conv, stride_conv, act_conv, 
                                        weights_dense, bias_dense, act_dense)
                                    )
                    
                    weights_dense[n_dense][i,j] -= 2*epsilon
            
                    f_minus = np.sum(activation(input_,
                                        weights_conv, stride_conv, act_conv, 
                                        weights_dense, bias_dense, act_dense)
                                    )
                    
                    weights_dense[n_dense][i,j] += epsilon
                                        
                    derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
            
            # bias                 
            for i in range(net.layers[n].bias.shape[1]):
            
                bias_dense[n_dense][:,i] += 2*epsilon
                
                f_plus = np.sum(activation(input_,
                                    weights_conv, stride_conv, act_conv, 
                                    weights_dense, bias_dense, act_dense)
                                )
                
                bias_dense[n_dense][:,i] -= 2*epsilon
        
                f_minus = np.sum(activation(input_,
                                    weights_conv, stride_conv, act_conv, 
                                    weights_dense, bias_dense, act_dense)
                                )
                
                bias_dense[n_dense][:,i] += epsilon
                                    
                derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
            
            n_dense += 1
                
        # convolutional layer
        elif net.layers[n].type == 'conv':
                                    
            # bias                 
            for i in range(net.layers[n].weights.shape[1]):
            
                weights_conv[n_conv][:,i] += 2*epsilon
                
                f_plus = np.sum(activation(input_,
                                    weights_conv, stride_conv, act_conv, 
                                    weights_dense, bias_dense, act_dense)
                                )
                
                weights_conv[n_conv][:,i] -= 2*epsilon
        
                f_minus = np.sum(activation(input_,
                                    weights_conv, stride_conv, act_conv, 
                                    weights_dense, bias_dense, act_dense)
                                )
                
                weights_conv[n_conv][:,i] += epsilon
                                    
                derivative_by_def.append((f_plus-f_minus)/(2*epsilon))
            
            n_conv += 1

    # evaluate error
    derivative_by_def = np.array(derivative_by_def).flatten()
    
    error = np.linalg.norm((derivative_by_calc-derivative_by_def)/np.linalg.norm(derivative_by_calc))
    
    print("Error: ", error)        
        
