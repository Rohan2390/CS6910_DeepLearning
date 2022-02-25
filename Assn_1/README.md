#CS6910 Deep Learning Assignment 1

This code implements Dense Neural Network with Back Propagation using various different optimizers

### Files Description

- dataPreparation.py => Downloads Fashion MNIST data and prepares for training.
- NeuralNetwork.py => Contains class for Neural Network
- optimizers.py => Contains different optimizers like SGD,ADAM,RMSPROP etc.
- Utils.py => Contains Utility functions used like Activation Functions and Losses.
- train.py => Class wrapper used to train by providing pipeline for training.
- DL Assigment 1 WandB.ipynb => Notebook used to train sweeps on WandB.

This code implemented to run both WandB and normal Training pipelines.

### For WandB Sweep run
DL Assigment 1 WandB.ipynb files is used to run on WandB.

It contains **wandb_config** which is used as config for different hyper-parameters to give sweep, also **sweep_config** is used to config sweep strategies.

### For Normal Non-Wandb run
train.py contains config which controls normal runs hyper-parameters.
```
python train.py
 ```
After changing config, above command will run non-wandb run.

### Config Keys
All of these keys are mandatory
```
numInputs: Number of input neurons
numHiddenLayers: Number of hidden layers
numHiddenLayersNeuron: Number of neurons in hidden layer
numOutputOfNeuron: number of neurons in output layer
actFun: Activation Function to used should be one of ReLU,Sigmoid,Tanh
loss: Loss function to use should be one of CrossEntropyLoss,SquaredLoss
optimizer: Optimizer to use should be one of SGD,MOMENTUM,NESTEROV,RMSPROP,ADAM,NADAM
bs: Batch Size to run on
epochs: epochs to run
```
