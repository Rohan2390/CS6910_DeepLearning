from dataPreparation import PrePareMNISTData
from NeuralNetwork import NeuralNetwork
from tqdm import tqdm
from random import shuffle
from Utils import *
from optimizers import *
import wandb

class Trainer:

    def __init__(self,config):

        self.nn = NeuralNetwork(config['numInputs'],
                                [config['numHiddenLayersNeuron']]*config['numHiddenLayers'],
                                config['numOutputOfNeuron'],
                                [actFunDict[config['actFun']]]*config['numHiddenLayers']+[Softmax],
                                config['xaviers']
                                )
        self.loss = lossFunDict[config['loss']]()
        self.optimArgs = config['optimArgs']
        self.optimArgs.update({'nn': self.nn})
        self.optimizer = optDict[config['optimizer']](**self.optimArgs)

        self.bs = config['bs']
        self.epochs = config['epochs']

    def run(self,xTrain,yTrain,xVal,yVal,wandbLog=False):

        indices = list(range(len(xTrain)))

        for epoch in tqdm(range(self.epochs)):

            shuffle(indices)
            cLoss = 0

            for batch in tqdm(range(len(xTrain)//self.bs)):

                if isinstance(self.optimizer,(NESTEROV,NADAM)):
                    self.optimizer.partialUpdate()

                batchI = indices[batch*self.bs:(batch+1)*self.bs]
                xBatch = xTrain[batchI]
                yBatch = yTrain[batchI]

                _,_,yPreds = self.nn.ForwardProp(xBatch)

                self.loss.CalculateLoss(yBatch,yPreds)
                self.nn.BackProp(self.loss)
                self.optimizer.update()
                cLoss+=self.loss.lossVal

            _,_,valPreds = self.nn.ForwardProp(xVal)
            self.loss.CalculateLoss(yVal,valPreds)

            if wandbLog:
                wandb.log({"Train Loss": cLoss / batch,"Val Loss":self.loss.lossVal,
                           "Val Acc":np.mean(np.argmax(valPreds,axis=1)==np.argmax(yVal,axis=1)),
                           "epoch":epoch
                           })
            else:
                print(f'For epoch:{epoch}')
                print('Train Loss:',cLoss/batch)
                print(f'Val Loss:{self.loss.lossVal}, Val acc:{np.mean(np.argmax(valPreds,axis=1)==np.argmax(yVal,axis=1))},{len(yVal)}')


if __name__ == '__main__':
    config = {
        'numInputs':28*28,
        'numHiddenLayers':3,
        'numHiddenLayersNeuron':128,
        'numOutputOfNeuron':10,
        'actFun':'ReLU',
        'loss':'CrossEntropyLoss',
        'optimizer':'ADAM',
        'optimArgs':{'lr':0.001,'wd':0},
        'xaviers':True,
        'bs':64,
        'epochs':10,
    }

    (xTrain,yTrain),(xVal,yVal),(xTest,yTest) = PrePareMNISTData()
    trainer = Trainer(config)
    trainer.run(xTrain,yTrain,xVal,yVal,wandbLog=False)
