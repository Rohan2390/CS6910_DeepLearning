import numpy as np
"""
# This file utils.py contains all the activation functions like sigmoid,softmax,relu ,tanh.
# The respective derivatives are also calculated.
"""


def Sigmoid(x):
    """
    :param x: input(images from fashion mnist)
    :return: sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    """
    :param x: input(images from fashion mnist)
    :return: derivative of sigmoid function
    """
    return x * (1 - x)

def Softmax(x):
    """
    :param x: input(images from fashion mnist)
    :return: softmax function
    """
    exp = np.exp(x)
    return exp / np.sum(exp,axis=1).reshape(-1,1)

def Tanh(x):
    """
    :param x: input(images from fashion mnist)
    :return: tan hyperbolic function
    """
    return np.tanh(x)

def TanhDerivative(x):
    """
    :param x: input(images from fashion mnist)
    :return: derivative of tan hyperbolic function
    """
    return (1-x**2)

def ReLU(x):
    """
    :param x: input(images from fashion mnist)
    :return: relu function
    """
    x[x<0]=0
    return x

def ReLUDerivative(x):
    """
    :param x: input(images from fashion mnist)
    :return:
    """
    x[x>0]=1
    x[x<=0]=0
    return x


"""
cross entropy loss and squared loss functions are created as separate class.
Also their respective derivatives are calculated
"""

class CrossEntropyLoss:

    def __init__(self):
        pass

    def CalculateLoss(self,yTrue,yPreds):
        '''
        :param yTrue: actual label
        :param yPreds: predicted label
        :return: cross entropy loss and gradient
        '''

        self.lossVal = np.mean(-1*yTrue*np.log(yPreds))
        self.lossGradientVal = yPreds-yTrue
        self.lossGradientVal = self.lossGradientVal.reshape((-1,1,len(yTrue[0])))

class SquaredLoss:

    def __init__(self):
        pass

    def CalculateLoss(self,yTrue,yPreds):
        '''
        :param yTrue: actual label
        :param yPreds: predicted label
        :return: mean squared error loss and gradient
        '''
        self.lossVal = np.mean(0.5*(yTrue-yPreds)**2)/2

        yTerm = (yPreds-yTrue)*yPreds
        iTerm = np.concatenate([np.expand_dims(yTrue,axis=2)]*len(yTrue[0]),axis=2)-np.concatenate([np.expand_dims(yPreds,axis=1)]*len(yTrue[0]),axis=1)

        self.lossGradientVal = np.matmul(np.expand_dims(yTerm,axis=1),iTerm)

actFunDerivative = {Sigmoid:SigmoidDerivative,
              ReLU:ReLUDerivative,
              Tanh:TanhDerivative
              }

actFunDict = {
    'ReLU':ReLU,
    'Sigmoid':Sigmoid,
    'Tanh':Tanh
}

lossFunDict = {
    'CrossEntropyLoss':CrossEntropyLoss,
    'SquaredLoss':SquaredLoss
}
