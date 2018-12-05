# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:46:58 2018

@author: Emanuele

Derivatives of loss functions: provides both the loss functions and the 
 dictionary to easy call them.
"""

import numpy as np


def l1(target, prediction):
    
    return np.sign(target-prediction);


def l2(target, prediction):
    
    return -2*(target-prediction)


dict_derivatives_losses = { 'L1': l1,
                           'L2': l2
                           }
