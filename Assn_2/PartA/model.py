from keras.models import Sequential
from keras.layers import Dense,Conv2D,Activation,MaxPooling2D,Flatten,BatchNormalization,Dropout

#Gives List of filters using number and org
def getFilterList(f, org):

    if org=='same':
        return [f]*5
    elif org=='doubling':
        return [(2**i)*f for i in range(5)]
    elif org=='halving':
        return [f/(2**i) for i in range(5)]
    else:
        raise ValueError


class CNNModel:

    def __init__(self,config):

        self.model = Sequential()

        nFilters = getFilterList(config['filters'],config['filterorg'])
        activationFunctions = [i for i in config['activationFunctions'].split(',')]

        #Make sure 6 activation functions are there
        if len(activationFunctions)!=6:
            raise ValueError

        #Add Conv Blocks
        for i in range(5):

            self.model.add(Conv2D(nFilters[i], (config['filtersize'], config['filtersize']),
                                  input_shape=(config['imageSize'],config['imageSize'],3)))
            self.model.add(Activation(activationFunctions[i]))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(config['maxPoolFilterSize'], config['maxPoolFilterSize'])))
            self.model.add(Dropout(config['dropout']))

        #Add Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(config['denseNeurons'], activation=activationFunctions[-1]))
        self.model.add(Dense(10, activation='softmax'))
        self.model.summary()

    #Compile Model
    def compile(self,**kwargs):
        self.model.compile(**kwargs)

    #Fit Model
    def fit(self,**kwargs):
        self.model.fit(**kwargs)

