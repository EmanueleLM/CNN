# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:38:00 2018

@author: Emanuele

Neural network's module: instantiate and train a convolutional neural network
"""

import layer_convolutional as layer_conv
import layer_dense as layer_dense
import utils as utils

import sys as sys
import numpy as np
import zlib as zlib

class NN(object):
    
    """
    Create a neural network as a dictionary composed by blocks of affine transformations.
    Each block is composed by ...
    Takes as input:
        blocks:list, specifies the affine transformations the net
               implements;
        compress:boolean, specifies if the network is compressed after creation to reduce
                   size on disk. Moreover, the model can be saved on disk and 
                   exported easily. This may cause overheads due to decompression
                   during learning phase;
        nn_id:int, the id of the neural network.
    """
    def __init__(self, 
                 blocks, 
                 compress=False,
                 nn_id=None):
        
        self.layers = list()
        self.n_inputs = blocks['n_inputs']
        self.nn_id = np.random.randint(0, 1e+9) if nn_id is None else nn_id
           
        # append dense and convolutional layers to the net
        blocks = blocks['layers']
        for i in range(blocks.__len__()):
                        
            if blocks[i]['type'] == 'dense':
                
                # delegate the output's shape calculation to the net:
                if blocks[i]['shape'][0] is None:
                        
                    if self.layers.__len__() != 0:
                        output_len = blocks[i]['shape'][1] 
                        
                        blocks[i]['shape'] = (self.layers[i-1].output_len, output_len)
                                             
                    else:
                        output_len = blocks[i]['shape'][1]
                        blocks[i]['shape'] = (self.n_inputs, output_len)
                    
                    layer = layer_dense.Dense(shape=blocks[i]['shape'],
                                              activation=blocks[i]['activation'],
                                              output_len=output_len)
                    self.layers.append(layer)
                    
                else:
                    
                    layer = layer_dense.Dense(shape=blocks[i]['shape'],
                          activation=blocks[i]['activation'])
                    self.layers.append(layer)
                    
            elif blocks[i]['type'] == 'conv':
                
                if self.layers.__len__() != 0:
                    output_len = utils.conv_shape(self.layers[-1].output_len,
                                                  blocks[i]['shape'][1],
                                                  blocks[i]['stride'])
                else:
                    output_len = utils.conv_shape(self.n_inputs, blocks[i]['shape'][1], 
                                                  blocks[i]['stride'])

                layer = layer_conv.Conv(shape=blocks[i]['shape'], 
                                     stride=blocks[i]['stride'], 
                                     activation=blocks[i]['activation'],
                                     output_len=output_len)
                self.layers.append(layer)                   
        
            else:
                
                print("error: no type for the layer has been specified", file=sys.stderr)
                pass
                    
        if compress is True:
            
            return NN_Compressed(nn=self, drop=True)
    
    """    
     Initialize the parameters of the net.
     Takes as input:
         parameters:list, is a list, one for each layer, that specifies the method and 
          parameters to be used for that specific layer. If just one element is specified,
          all the layer are initialized with that method/arguments.
          Each entry in the parameter is a list composed by a method that is used to
          initialize the i-th layer, and the parameters involved in the initialization.
          Take as reference the module 'parameters_init.py'.
          An example for a 3 layers neural network with the same initialization
           method for all the layers will define parameters = ['uniform', p_1, p_2]
           where p_1, p_2 are the parameters involved in the initialization: upper
           and lower bounds for the unifrom function, in this case.
           An example with 3 different initialiations is 
            parameters = [['random'], ['uniform', -1., 1.], ['random']].
    """
    def init_parameters(self, parameters):
        
        if type(parameters[0]) != type(list()):
            
            iter_ = [parameters for _ in range(len(net_blocks['layers']))]
        
        else:
            
            iter_ = parameters
                        
        for i in range(len(iter_)):
            
            if len(iter_[i]) > 1:
            
                self.layers[i].init_parameters(iter_[i][0], iter_[i][1:])
            
            else:

                self.layers[i].init_parameters(iter_[i][0])
            
            
            
        
    # activation of the neural network
    def activation(self, input_):
        
        tmp = input_
        
        for layer in self.layers:
            
            tmp = layer.activation(tmp)
        
        return tmp
            
def NN_Compressed(object):
    
    def __init__(self, 
                 nn, 
                 drop=True):
        
        self.nn = zlib.compress(nn)
        
        if drop is True:
            
            del nn

# Code demonstration by example: 
#  abilitate this snippet iff you are sure what you are doing.
verbose = True

if verbose is True:
    
    net_blocks = {'n_inputs': 1000, 
                  'layers': [
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 5), 'stride': 2}, 
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 4), 'stride': 2},
                          {'type': 'conv', 'activation': 'relu', 'shape': (1, 3), 'stride': 3},
                          {'type': 'dense', 'activation': 'linear', 'shape': (None, 30)},
                          {'type': 'dense', 'activation': 'linear', 'shape': (None, 2)}
                          ]
                  }
    
    net = NN(net_blocks)
    
    # initialize the parameters
    net.init_parameters(['uniform', 0., 2.])
    
    # activate the net for a random input
    output_ = net.activation(np.random.rand(1, 1000))
    
            