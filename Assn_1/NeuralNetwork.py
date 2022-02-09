import numpy as np

np.random.seed(0)


class NeuralNetwork():

  def __init__(self, numInputs, numHiddenLayers, numOutputs, actFun):

    self.numInputs = numInputs
    self.numHiddenLayers = numHiddenLayers
    self.numOutputs = numOutputs
    self.actFun = actFun

    # Number of neurons in layers
    layers = [numInputs] + numHiddenLayers + [numOutputs]
    self.numOfLayers = len(layers)

    # Initialising random bias & weights
    self.weights = []
    self.bias = []
    for i in range(len(layers) - 1):
      self.weights.insert(i, (np.random.rand(layers[i + 1], layers[i])))
      self.bias.insert(i, (np.random.rand(layers[i + 1], 1)))

  def ForwardProp(self, X):
    self.a = {}
    self.h = {}
    self.h[0] = X
    self.a[0] = X

    for i, (w, b, af) in enumerate(zip(self.weights, self.bias, self.actFun)):
      self.a[i + 1] = w.dot(self.h[i]) + b
      self.h[i + 1] = af(self.a[i])

    return self.h, self.a, self.h[i + 1]
