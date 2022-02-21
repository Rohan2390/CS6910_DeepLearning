class SGD:

    def __init__(self,nn,lr=0.001):

        self.nn = nn
        self.lr = lr

    def update(self):

        for i in range(self.nn.num_of_layers-1):

            self.nn.weights[i] = self.nn.weights[i] - self.lr*self.nn.wUpdate[i]
            self.nn.bias[i] = self.nn.bias[i] - self.lr*self.nn.bUpdate[i]
            
           
class MOMEMTUM:

    def __init__(self,nn,gamma=0.9,lr=0.001):

        self.nn = nn
        self.lr = lr
        self.gamma = gamma
        
        self.lastUpdateW =  [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB =  [np.zeros_like(i) for i in nn.weights]


    def update(self):

        for i in range(self.nn.num_of_layers-1):

            
            self.lastUpdateW[i] = (gamma * self.lastUpdateW[i]) + (self.lr * self.nn.wUpdate[i])
            self.nn.weights[i] = self.nn.weights[i] - self.lastUpdateW[i] 
            
            
            self.lastUpdateB[i] = (gamma * self.lastUpdateB[i]) + (self.lr * self.nn.bUpdate[i])
            self.nn.bias[i] = self.nn.bias[i] - self.lastUpdateB[i]          
