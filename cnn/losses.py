# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:30:29 2018

@author: Emanuele

Loss functions: provides both the loss functions and the dictionary to easy 
 call them.
"""

import numpy as np


def l1(target, prediction):
    
    return np.abs(target-prediction);


def l2(target, prediction):
    
    return (target-prediction)**2


# loss for testing purposes: just return the prediction
def __none(target, prediction):
    
    return prediction


dict_derivatives_losses = { 'L1': l1,
                            'L2': l2,
                            'none': __none 
                           }
