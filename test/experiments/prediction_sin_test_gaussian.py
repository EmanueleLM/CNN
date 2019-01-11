# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 09:30:44 2019

@author: Emanuele
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats
import sys

sys.path.append('../../cnn')
sys.path.append('../experiments/data_utils')

import nn as nn
import utils_topix as utils


if __name__ == '__main__':
    
    net_blocks = {'n_inputs': 1, 
                  'layers': [ 
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 10)},                    
                          {'type': 'dense', 'activation': 'relu', 'shape': (None, 1)}
                          ]
                  }
    
    # create the net    
    net = nn.NN(net_blocks)
    
    # initialize the parameters
    net.init_parameters(['uniform', -.1e-1, 1e-0])

    # create the batches from topix dataset
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = utils.generate_batches(
                                                              filename='data/sin.csv', 
                                                              window=net.n_inputs, mode='validation', 
                                                              non_train_percentage=.3,
                                                              val_rel_percentage=.9,
                                                              normalize=True,
                                                              time_diffference=True)       

    epochs_train = 25
       
    # train
    for e in range(epochs_train):
        
        print("Training epoch ", e+1)
        
        for (input_, target) in zip(X_train, Y_train):
            
            # format input and prediction
            input_ = input_[np.newaxis,:]
            target = np.array([target])[np.newaxis,:]
                      
            # activate and caluclate derivative for each layer, given the input
            net.activation(input_, accumulate=True)
            net.derivative(None)
                        
            # execute the backpropagation with the input that has been memorized previously
            net.backpropagation(target=target, 
                                loss='L2', 
                                optimizer='sgd', 
                                l_rate=1e-2,
                                update=True)
                        
    # validation: calculate error and estimate its mean and variance
    errors_valid = np.zeros(shape=len(X_valid))
    i = 0
    
    for (input_, target) in zip(X_valid, Y_valid):
        
        
        # format input and prediction
        input_ = input_[np.newaxis,:]
        target = np.array([target])[np.newaxis,:]
        
        net.activation(input_, accumulate=True)
        
        errors_valid[i] = net.output - target
        
        i += 1
    
    mean_valid, std_valid = (errors_valid.mean(), errors_valid.std())
    
    # test   
    p_anomaly_test = np.zeros(shape=len(X_test))
    predictions = np.zeros(shape=len(X_test))
    errors_test = np.zeros(shape=len(Y_test))
    threshold = utils.gaussian_pdf(mean_valid-2*std_valid, mean_valid, std_valid)
    i = 0
     
    for (input_, target) in zip(X_test, Y_test):
        
        # format input and prediction
        input_ = input_[np.newaxis,:]
        target = np.array([target])[np.newaxis,:]
        
        net.activation(input_, accumulate=True)
        prediction = net.output
        
        predictions[i] = prediction
        errors_test[i] = scistats.norm.ppf(prediction-target, mean_valid, std_valid)
        anomalies = np.argwhere(errors_test < threshold)   
                
        i += 1
        
    # plot results
    fig, ax1 = plt.subplots()

    # plot data series
    ax1.plot(Y_test[:len(Y_test)-net.n_inputs+1], 'b', label='index')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('TOPIX')

    # plot predictions
    ax1.plot(predictions, 'r', label='prediction')
    ax1.set_ylabel('Change Point')
    plt.legend(loc='best')
    
    for i in anomalies:
        
        if i <= len(Y_test)-net.n_inputs:
            
            plt.axvspan(i, i+1, color='yellow', alpha=0.5, lw=0)

    fig.tight_layout()
    plt.show()
    
    import random; print("\nTen couples of (prediction, target):\n",
                         random.sample(set(zip(predictions, Y_test)), 10))    
    
    print("\nTotal error on ", len(predictions), "points is ", 
          np.linalg.norm(Y_test[:len(Y_test)-net.n_inputs+1]-predictions))
    