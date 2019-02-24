# Convolutional Neural Networks for Time Series Analysis
## Package to instantiate 1-D CNNs and train them on signals datasets

## Things implemented so far: ##

- customizable deep net (specify with one line of code the net, from the topology to the activations);
  - all from backpropagation to check gradient routine depends only on numpy;
  - easy to extend activations, derivatives, parameters initialization etc.
  

TODO:

- refactor code and package's name to handle 2D/3D images;

- topologies that mix layers of convolutionals and dense type: so far it is possible to specify
  n>0 convolutional layers followed by m>0 dense layers.

## Create a double-layer CNN followed by a final double-dense layer

```python
import nn as nn  # nn.py from this package
import numpy as np

# define networks layers
net_blocks = {'n_inputs': 50, 
              'layers': [
                      {'type': 'conv', 'activation': 'leaky_relu', 'shape': (15, 2), 'stride': 3},                      
                      {'type': 'conv', 'activation': 'leaky_relu', 'shape': (30, 2), 'stride': 3},                      
                      {'type': 'dense', 'activation': 'relu', 'shape': (None, 75)},                    
                      {'type': 'dense', 'activation': 'relu', 'shape': (None, 1)}
                      ]
              }
    
# create the net    
net = nn.NN(net_blocks)
    
# initialize the parameters
net.init_parameters(['uniform', -.1e-1, 1e-1])
```

Now you can check one of the test in the /test/experiments forlder to see how to run this model against a dataset.