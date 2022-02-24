from dataPreparation import PrePareMNISTData
from NeuralNetwork import NeuralNetwork
from tqdm import tqdm
from random import shuffle
from Utils import *
from optimizers import *

class Trainer:

    def __init__(self,config):

        self.nn = NeuralNetwork(config['numInputs'],config['numHiddenLayers'],config['numOutputs'],config['actFun'],config['xaviers'])
        self.loss = config['loss']()
        self.optimArgs = config['optimArgs']
        self.optimArgs.update({'nn': self.nn})
        self.optimizer = config['optimizer'](**self.optimArgs)

        self.bs = config['bs']
        self.epochs = config['epochs']
        #self.printVal = config['printVal']

    def run(self,xTrain,yTrain,xVal,yVal):

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

            print('Train Loss:',cLoss/batch)

            _,_,valPreds = self.nn.ForwardProp(xVal)
            self.loss.CalculateLoss(yVal,valPreds)

            print(f'Val Loss:{self.loss.lossVal}, Val acc:{np.mean(np.argmax(valPreds,axis=1)==np.argmax(yVal,axis=1))},{len(yVal)}')


if __name__ == '__main__':
    config = {
        'numInputs':28*28,
        'numHiddenLayers':[128,128,128],
        'numOutputs':10,
        'actFun':[ReLU,ReLU,ReLU,Softmax],
        'loss':CrossEntropyLoss,
        'optimizer':ADAM,
        'optimArgs':{'lr':0.001,'wd':0},
        'xaviers':True,
        'bs':64,
        'epochs':10,
    }

    (xTrain,yTrain),(xVal,yVal),(xTest,yTest) = PrePareMNISTData()
    trainer = Trainer(config)
    trainer.run(xTrain,yTrain,xVal,yVal)
