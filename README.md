## Shallow Graph Convolutional Neural Network
This is the re-implementation of the GCN model, which is from the paper
`SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NEURAL NETWORK` by Thomas N. Kipf et al.
This implementation is a simplified version adapted from https://github.com/tkipf/gcn by removing dense and Chebyshev models.
Moreover, it supports multiple graph convolutional layers just by setting parameters of `GCN` object.
According to my experiments, GCN with more than two layers doesn't seem to improve the classification performance.

## Requirements
- python 3+
- tensorflow 1.6+

## How to run
To train a default GCN model using the `cora` dataset: 

`python main.py`

Optionally, you can change dataset, learning rate, the number of training epochs, dropout, weight decay and early stopping constant. For example:

`python main.py --dataset citeseer --learning_rate 0.001`

You can also change the number of hidden layers and their dimensionality:

`python main.py --hidden_dimensions=64,64`
