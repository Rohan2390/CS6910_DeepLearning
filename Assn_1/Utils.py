import numpy as np

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    return x * (1 - x)

def Softmax(x):
    return np.exp(x) / sum(np.exp(x))

def Tanh(x):
    return np.tanh(x)

def TanhDerivative(x):
    return (1-x**2)

def ReLU(x):
    x[x<0]=0
    return x

def ReLUDerivative(x):
    x[x>0]=1
    x[x<=0]=0
    return x

actFunDerivative = {Sigmoid:SigmoidDerivative,
              ReLU:ReLUDerivative,
              Tanh:TanhDerivative
              }

class CrossEntropyLoss:

    def __init__(self):
        pass

    def CalculateLoss(self,yTrue,yPreds):
        self.lossVal = np.mean(-1*yTrue*np.log(yPreds))
        self.lossGradientVal = yPreds-yTrue
        self.lossGradientVal = self.lossGradientVal.reshape((-1,1,len(yTrue[0])))

class SquaredLoss:

    def __init__(self):
        pass

    def CalculateLoss(self,yTrue,yPreds):
        self.lossVal = np.mean(0.5*(yTrue-yPreds)**2)
        self.lossGradientVal = yPreds-yTrue
        self.lossGradientVal = self.lossGradientVal.reshape((-1,1,len(yTrue[0])))