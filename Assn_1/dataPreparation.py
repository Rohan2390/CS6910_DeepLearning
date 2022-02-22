from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

#Normalization may be added in future

def PrePareMNISTData():

    (xTrain,yTrain),(xTest,yTest) = fashion_mnist.load_data()

    PlotData(xTrain,yTrain)

    xTrain = xTrain.reshape(-1,28*28) #Add this to config
    xTest = xTest.reshape(-1,28*28) #Add this to config

    yTrain = np.eye(10)[yTrain]
    yTest = np.eye(10)[yTest]

    (xTrain,yTrain),(xVal,yVal) = splitData(xTrain,yTrain)

    return (xTrain/255.0,yTrain),(xVal/255.0,yVal),(xTest/255,yTest)

def splitData(x,y,percent=10):

    valIndex = np.random.randint(0,len(x),size=int(0.1*len(x)))
    trainIndex = [i for i in range(len(x)) if i not in valIndex]

    return (x[trainIndex],y[trainIndex]),(x[valIndex],y[valIndex])


def PlotData(x,y):

    classes = np.unique(y)

    fig,ax = plt.subplots(1,len(classes),figsize=(20,40))

    for i in classes:
        ax[i].imshow(x[y==i][0])

if __name__=='__main__':
    PrePareMNISTData()
    plt.show()