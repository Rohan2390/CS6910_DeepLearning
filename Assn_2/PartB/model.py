from keras.applications.efficientnet_v2 import EfficientNetV2B0
from keras.models import Sequential
from keras.layers import Dense

baseModelDict = {'EffnetV2B0':EfficientNetV2B0}

class TLModel:

    def __init__(self,baseModel='EffnetV2B0'):

        self.baseModel = baseModelDict[baseModel](
            include_top=False,
            classes=10,
            pooling='max'
        )
        self.baseModel.trainable = False

        self.model = Sequential()
        self.model.add(self.baseModel)
        self.model.add(Dense(10,activation='softmax'))

        self.model.get_config()

    def stopLayers(self,percent):
        pass

    def compile(self,**kwargs):
        self.model.compile(**kwargs)

    def fit(self,**kwargs):
        self.model.fit(**kwargs)

