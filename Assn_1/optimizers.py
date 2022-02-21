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

class RMSPROP:

    def __init__(self,nn,beta=0.9,eps= 1e-8,lr=0.001):

        self.nn = nn
        self.lr = lr
        self.beta = beta
        self.eps = eps
         
        self.lastUpdateV =  [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB =  [np.zeros_like(i) for i in nn.weights]


    def update(self):

        for i in range(self.nn.num_of_layers-1):

            
            self.lastUpdateV[i] = beta * self.lastUpdateV[i] + (1 - beta) * self.nn.wUpdate[i]**2
            self.nn.weights[i] = self.nn.weights[i] - (self.lr * self.nn.wUpdate[i]) /( np.sqrt(self.lastUpdateV[i])+ eps)
            
            self.lastUpdateB[i] = beta * self.lastUpdateB[i] + (1 - beta) * self.nn.bUpdate[i]**2
            self.nn.weights[i] = self.nn.weights[i] - (self.lr * self.nn.bUpdate[i]) /( np.sqrt(self.lastUpdateB[i])+ eps)
            
