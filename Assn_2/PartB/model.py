from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import  InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from keras.models import Sequential
from keras.layers import Dense

#Base Model Dict
baseModelDict = {
    'InceptionResNetV2':InceptionResNetV2,
    'InceptionV3':InceptionV3,
    'ResNet50':ResNet50,
    'Xception':Xception
}

#TRansfer Learning Model
class TLModel:

    def __init__(self,baseModel='EffnetV2B0',epochs=10,pTrainLayers=0.1,denseNeurons=1000):

        self.epochs = epochs
        self.pTrainLayers = pTrainLayers #Percentage of layers to train by the end

        #Get Base Model for Transfer Learning
        self.baseModel = baseModelDict[baseModel](
            include_top=False,
            classes=10,
            pooling='max'
        )
        self.baseModel.trainable = False

        #Create Model
        self.model = Sequential()
        self.model.add(self.baseModel)
        self.model.add(Dense(denseNeurons,activation='relu'))
        self.model.add(Dense(10,activation='softmax'))

    def startLayers(self,currentEpoch):
        #Start Training of layers according to epch and p
        if currentEpoch!=0 and self.pTrainLayers!=0:
            for layer in self.baseModel.layers[int(-1*len(self.baseModel.layers)*(currentEpoch*self.pTrainLayers)/self.epochs):]:
                layer.trainable=True

    #Compile Model
    def compile(self,**kwargs):
        self.model.compile(**kwargs)

    #Fit model
    def fit(self,**kwargs):
        self.model.fit(**kwargs)

