# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:38:00 2018

@author: Emanuele

Neural network's module: instantiate and train a convolutional neural network
"""

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
        
        self.nn_id = np.random.randint(0, 1e+13) if nn_id is None else nn_id
        
        for b in blocks:
            pass;
            
        if compress is True:
            
            return NN_Compressed(nn=self, drop=True)
            
def NN_Compressed(object):
    
    def __init__(self, 
                 nn, 
                 drop=True):
        
        self.nn = zlib.compress(nn)
        
        if drop is True:
            
            del nn
            