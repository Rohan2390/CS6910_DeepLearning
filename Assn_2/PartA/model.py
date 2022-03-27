from keras.models import Sequential
from keras.layers import Dense,Conv2D,Activation,MaxPooling2D

class CNNModel:

    def __init__(self,denseNeurons=1000):

        self.model = Sequential()

        for i in range(5):

            self.model.add(Conv2D(16, (3, 3), input_shape=(256,256,3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dense(denseNeurons, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

    def compile(self,**kwargs):
        self.model.compile(**kwargs)

    def fit(self,**kwargs):
        self.model.fit(**kwargs)

