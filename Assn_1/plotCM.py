import numpy as np
import matplotlib.pyplot as plt
import wandb

classList = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

def plotCM(yPreds,yTrue,testAcc=None,wandbLog=False):

    cm = np.zeros((len(yPreds[0]),len(yPreds[0])))

    yPreds = np.argmax(yPreds,axis=1)
    yTrue = np.argmax(yTrue,axis=1)

    for i,j in zip(yPreds,yTrue):
        cm[i][j]+=1.0

    cm = cm/np.sum(cm,axis=1)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.matshow(cm,cmap=plt.cm.get_cmap('Blues'))
    ax.set(title=f'Confusion Matrix with Accuracy of {testAcc}',ylabel='Predicted Label',xlabel='True Label')

    if wandbLog:
        wandb.log({'Confusion Matrix': plt})
    else:
        plt.savefig('CM.png')