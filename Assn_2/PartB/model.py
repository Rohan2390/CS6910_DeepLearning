from keras.applications.efficientnet_v2 import EfficientNetV2B0
from keras.models import Sequential
from keras.layers import Dense

baseModelDict = {'EffnetV2B0':EfficientNetV2B0}

class TLModel:

    def __init__(self,baseModel='EffnetV2B0',epochs=10,pTrainLayers=0.1):

        self.epochs = epochs
        self.pTrainLayers = pTrainLayers

        self.baseModel = baseModelDict[baseModel](
            include_top=False,
            classes=10,
            pooling='max',
            include_preprocessing = True
        )
        self.baseModel.trainable = False

        self.model = Sequential()
        self.model.add(self.baseModel)
        self.model.add(Dense(10,activation='softmax'))

        self.model.get_config()

    def startLayers(self,currentEpoch):
        for layer in self.baseModel.layers[-1*len(self.baseModel.layers)*(currentEpoch*self.pTrainLayers)/self.epochs]:
            layer.trainable=True

    def compile(self,**kwargs):
        self.model.compile(**kwargs)

    def fit(self,**kwargs):
        self.model.fit(**kwargs)

