## Convolutional Neural Networks for Time Series Analysis
# Package to instantiate 1-D CNNs and train them on signals datasets

## Things implemented so far, 07/12/2018: ##

- customizable deep net (specify with one line of code the net, from the topology to the activations);
  - all from backpropagation to check gradient routine depend only on numpy;
  - easy to extend activations, derivatives, parameters initialization etc.
  
- kernels can be matrices (bidimensional);
  
- test on TOPIX dataset: anomaly detection with gaussian index variation.

TODO:

- topologies that mix layers of convolutionals and dense type: so far it is possible to specify
  n>0 convolutional layers followed by m>0 dense layers.