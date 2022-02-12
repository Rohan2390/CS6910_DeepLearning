class SGD:

    def __init__(self,nn,lr=0.001):

        self.nn = nn
        self.lr = lr

    def update(self):

        for i in range(self.nn.num_of_layers-1):

            self.nn.weights[i] = self.nn.weights[i] - self.lr*self.nn.wUpdate[i]
            self.nn.bias[i] = self.nn.bias[i] - self.lr*self.nn.bUpdate[i]