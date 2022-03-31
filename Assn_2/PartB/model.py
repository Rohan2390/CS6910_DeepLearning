from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import  InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from keras.models import Sequential
from keras.layers import Dense

baseModelDict = {
    'InceptionResNetV2':InceptionResNetV2,
    'InceptionV3':InceptionV3,
    'ResNet50':ResNet50,
    'Xception':Xception
}

class TLModel:

    def __init__(self,baseModel='EffnetV2B0',epochs=10,pTrainLayers=0.1,denseNeurons=1000):

        self.epochs = epochs
        self.pTrainLayers = pTrainLayers

        self.baseModel = baseModelDict[baseModel](
            include_top=False,
            classes=10,
            pooling='max'
        )
        self.baseModel.trainable = False

        self.model = Sequential()
        self.model.add(self.baseModel)
        self.model.add(Dense(denseNeurons,activation='relu'))
        self.model.add(Dense(10,activation='softmax'))

    def startLayers(self,currentEpoch):
        if currentEpoch!=0 and self.pTrainLayers!=0:
            for layer in self.baseModel.layers[int(-1*len(self.baseModel.layers)*(currentEpoch*self.pTrainLayers)/self.epochs):]:
                layer.trainable=True

    def compile(self,**kwargs):
        self.model.compile(**kwargs)

    def fit(self,**kwargs):
        self.model.fit(**kwargs)

