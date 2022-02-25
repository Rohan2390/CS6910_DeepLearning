from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import wandb

def PrePareMNISTData(wandbLog=False):
    """
    used for preprocessing,reshaping and normalisation
    splitting data to train and test
    :param wandbLog: if given true will login in wandb
    :return: normalised values of input and output train and test values
    """

    (xTrain,yTrain),(xTest,yTest) = fashion_mnist.load_data()

    PlotData(xTrain,yTrain,wandbLog=wandbLog)

    xTrain = xTrain.reshape(-1,28*28) #Add this to config
    xTest = xTest.reshape(-1,28*28) #Add this to config

    yTrain = np.eye(10)[yTrain]
    yTest = np.eye(10)[yTest]

    (xTrain,yTrain),(xVal,yVal) = splitData(xTrain,yTrain)

    return (xTrain/255.0,yTrain),(xVal/255.0,yVal),(xTest/255,yTest)

def splitData(x,y,percent=10):
    """
    :param percent: As asked in the question 10% of data is used for validation
    :return: validation data for input and output
    """

    valIndex = np.random.randint(0,len(x),size=int(percent/100*len(x)))
    trainIndex = [i for i in range(len(x)) if i not in valIndex]

    return (x[trainIndex],y[trainIndex]),(x[valIndex],y[valIndex])


def PlotData(x,y,wandbLog=False):
    """
    :param x: input(images from fashion mnist)
    :param y: output(labels)
    :param wandbLog: if true will plot in wandb

    The data is prepared and displayed (only the unique values)
    """


    classes = np.unique(y)

    fig,ax = plt.subplots(1,len(classes),figsize=(10,10))

    for i in classes:
        ax[i].imshow(x[y==i][0])

    if wandbLog:
        wandb.log({'Data Plot': plt})
    else:
        plt.savefig('DataPlot.png')

if __name__=='__main__':
    PrePareMNISTData()
    plt.show()
