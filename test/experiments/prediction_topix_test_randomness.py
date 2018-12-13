# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:53:07 2018

@author: Emanuele

Predict TOPIX index
"""

import numpy as np
import sys

sys.path.append('../../cnn')
sys.path.append('../experiments/data_utils')

import matplotlib.pyplot as plt
import nn as nn
import utils_topix as utils


if __name__ == '__main__':
    
    net_blocks = {'n_inputs': 25, 
                  'layers': [
                          {'type': 'conv', 'activation': 'leaky_relu', 'shape': (25, 2), 'stride': 2}, 
                          {'type': 'conv', 'activation': 'leaky_relu', 'shape': (55, 2), 'stride': 2}, 
                          {'type': 'dense', 'activation': 'tanh', 'shape': (None, 75)},                    
                          {'type': 'dense', 'activation': 'tanh', 'shape': (None, 1)}
                          ]
                  }
    
    # create the net    
    net = nn.NN(net_blocks)
    
    # initialize the parameters
    net.init_parameters(['uniform', .0, 1e-1])

    # create the batches from topix dataset
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = utils.generate_batches(
                                                              filename='data/Topix_index.csv', 
                                                              window=net.n_inputs, mode='validation', 
                                                              non_train_percentage=.3,
                                                              val_rel_percentage=.85)
    
    # normalize the dataset (max-min method)
    X_train = (X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
    X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
    X_valid = (X_valid-np.min(X_valid))/(np.max(X_valid)-np.min(X_valid))
    Y_train = (Y_train-np.min(Y_train))/(np.max(Y_train)-np.min(Y_train))
    Y_test = (Y_test-np.min(Y_test))/(np.max(Y_test)-np.min(Y_test))   
    Y_valid = (Y_valid-np.min(Y_valid))/(np.max(Y_valid)-np.min(Y_valid))    
    
    epochs_train = 10
       
    # train
    for e in range(epochs_train):
        
        print("Training epoch ", e+1)
        
        for (input_, target) in zip(X_test, Y_train):
            
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
        
        # backrpop after prediction
        net.derivative(None)                    
        net.backpropagation(target=target, 
                            loss='L2', 
                            optimizer='sgd', 
                            l_rate=1e-3,
                            update=True)
        
        errors_valid[i] = net.output - target
        
        i += 1
    
    mean_valid = errors_valid.mean()
    
    # test   
    p_anomaly_test = np.zeros(shape=len(X_test))
    predictions = np.zeros(shape=len(X_test))
    anomaly_chunk_size = 8
    bin_errors_test = np.zeros(shape=anomaly_chunk_size)
    anomalies = list()
    alpha = 5e-3 #  test significance

    i = 0
     
    for (input_, target) in zip(X_test, Y_test):
        
        # format input and prediction
        input_ = input_[np.newaxis,:]
        target = np.array([target])[np.newaxis,:]
        
        net.activation(input_, accumulate=True)
        prediction = net.output
        
        predictions[i] = prediction
        bin_errors_test[i%anomaly_chunk_size] = (0 if (prediction-target) >= mean_valid else 1)
        
        # test randomness of the prediciton: every chunk of anomaly_chunk_size
        #  points is considered an anomaly if the related statistic supports 
        #  the (null) hypotesis
        if (i % anomaly_chunk_size) == 0 and i > 0:
            
            test_result = utils.random_test(bin_errors_test, alpha)
            bin_errors_test *= 0 #  reset the errors' vector for the next step
            
            # append the anomalies' indices
            if test_result is True:
                
                anomalies.append(i); anomalies.append(i-1); anomalies.append(i-2)
                        
        i += 1    
        
    # plot results
    fig, ax1 = plt.subplots()

    # plot data series
    ax1.plot(Y_test[:-net.n_inputs+1], 'b', label='index')
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
          np.linalg.norm(Y_test[:-net.n_inputs+1]-predictions))
    