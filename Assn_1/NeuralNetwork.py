import numpy as np
from Utils import actFunDerivative

np.random.seed(0)


class NeuralNetwork():

  def __init__(self, numInputs, numHiddenLayers, numOutputs, actFun , xaviers = False):
    """

    :param numInputs: consists the number of inputs
    :param numHiddenLayers: consists the number of hidden layers
    :param numOutputs: consists the number of outputs
    :param actFun: activation function
    :param xaviers: initialisation type

    """

    self.numInputs = numInputs
    self.numHiddenLayers = numHiddenLayers
    self.numOutputs = numOutputs
    self.actFun = actFun
    self.xaviers = xaviers

    # Number of neurons in layers
    layers = [numInputs] + numHiddenLayers + [numOutputs]
    self.numOfLayers = len(layers)

    # Initialising random bias & weights
    self.weights = []
    self.bias = []
    for i in range(len(layers) - 1): 
      if self.xaviers:
        self.weights.insert(i, (np.random.normal(0,(1/layers[i])**0.5,(layers[i], layers[i+1]))))
        self.bias.insert(i, (np.zeros((layers[i + 1],))))

      else:
        self.weights.insert(i, (np.random.uniform(-0.2,0.2,(layers[i], layers[i+1]))))
        self.bias.insert(i, (np.random.uniform(-0.2,0.2,(layers[i + 1],))))

  def ForwardProp(self, X):
    """
    This function takes image as input and returns output of its forward propagation.
    :param X: image input
    :return: preactivation function (a) and activation function (h)
    """
    self.a = {}
    self.h = {}
    self.h[0] = X
    self.a[0] = X

    for i, (w, b, af) in enumerate(zip(self.weights, self.bias, self.actFun)):
      self.a[i + 1] = np.matmul(self.h[i],w) + b
      self.h[i + 1] = af(self.a[i+1])

    return self.h, self.a, self.h[i + 1]

  def BackProp(self,loss):
    """
    This function takes loss as input and calculates gradient and stores them as instance variable.
    :param loss: takes the repective loss function(CEL OR MSE)
    :return: gradients from output layers,hidden layers and input layers(both activation and pre activation)
    """

    aGrad = loss.lossGradientVal
    self.wUpdate = {}
    self.bUpdate = {}

    for i in range(self.numOfLayers-2,-1,-1):
      self.wUpdate[i] = np.mean(aGrad*np.expand_dims(self.h[i],axis=2),axis=0)
      self.bUpdate[i] = np.mean(aGrad,axis=(0,1))
      
      #only for non input layers
      if i>0:
        hGrad = np.matmul(aGrad,self.weights[i].T)
        aGrad = hGrad*actFunDerivative[self.actFun[i-1]](np.expand_dims(self.a[i],axis=1))
