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

def CrossEntropyLoss(yTrue,yPreds):
    return np.sum(-1*yTrue*np.log(yPreds))

def SquaredLoss(yTrue,yPreds):
    return 0.5*(yTrue-yPreds)**2