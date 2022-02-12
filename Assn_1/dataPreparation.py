from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

#Normalization may be added in future

def PrePareMNISTData():

    (xTrain,yTrain),(xTest,yTest) = fashion_mnist.load_data()

    PlotData(xTrain,yTrain)

    xTrain = xTrain.reshape(-1,28*28) #Add this to config
    xTest = xTest.reshape(-1,28*28) #Add this to config

    (xTrain,yTrain),(xVal,yVal) = splitData(xTrain,yTrain)

    return (xTrain,yTrain),(xVal,yVal),(xTest,yTest)

def splitData(x,y,percent=10):

    valIndex = np.random.randint(0,len(x),size=int(0.1*len(x)))
    trainIndex = [i for i in range(len(x)) if i not in valIndex]

    return (x[trainIndex],y[trainIndex]),(x[valIndex],y[valIndex])


def PlotData(x,y):

    classes = np.unique(y)

    fig,ax = plt.subplot(1,len(classes),figsize=(20,40))

    for i in classes:
        ax[i].imshow(x[y[y==classes]][0])