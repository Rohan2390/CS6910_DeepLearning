import numpy as np
import math

class SGD:

    def __init__(self, nn, lr=0.001):
        self.nn = nn
        self.lr = lr

    def update(self):
        for i in range(self.nn.numOfLayers - 1):
            self.nn.weights[i] = self.nn.weights[i] - self.lr * self.nn.wUpdate[i]
            self.nn.bias[i] = self.nn.bias[i] - self.lr * self.nn.bUpdate[i]

class MOMEMTUM:

    def __init__(self, nn, gamma=0.9, lr=0.001):
        self.nn = nn
        self.lr = lr
        self.gamma = gamma

        self.lastUpdateW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB = [np.zeros_like(i) for i in nn.weights]

    def update(self):
        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateW[i] = (self.gamma * self.lastUpdateW[i]) + (self.lr * self.nn.wUpdate[i])
            self.nn.weights[i] = self.nn.weights[i] - self.lastUpdateW[i]

            self.lastUpdateB[i] = (self.gamma * self.lastUpdateB[i]) + (self.lr * self.nn.bUpdate[i])
            self.nn.bias[i] = self.nn.bias[i] - self.lastUpdateB[i]


class NESTEROV:

    def __init__(self, nn, gamma=0.9, lr=0.001):
        self.nn = nn
        self.lr = lr
        self.gamma = gamma

        self.lastUpdateW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB = [np.zeros_like(i) for i in nn.weights]

    def partialupdate(self):
        wahead = self.nn.w - self.gamma * update
        self.originalw = self.nn.w
        self.nn.w = wahead

    def update(self):
        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateW[i] = (self.gamma * self.lastUpdateW[i]) + (self.lr * self.nn.wUpdate[i])
            self.nn.weights[i] = self.nn.weights[i] - self.lastUpdateW[i]

            self.lastUpdateB[i] = (self.gamma * self.lastUpdateB[i]) + (self.lr * self.nn.bUpdate[i])
            self.nn.bias[i] = self.nn.bias[i] - self.lastUpdateB[i]


class RMSPROP:

    def __init__(self, nn, beta=0.9, eps=1e-8, lr=0.001):
        self.nn = nn
        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.lastUpdateV = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB = [np.zeros_like(i) for i in nn.weights]

    def update(self):
        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateV[i] = self.beta * self.lastUpdateV[i] + (1 - self.beta) * self.nn.wUpdate[i] ** 2
            self.nn.weights[i] = self.nn.weights[i] - (self.lr * self.nn.wUpdate[i]) / (
                        np.sqrt(self.lastUpdateV[i]) + self.eps)

            self.lastUpdateB[i] = self.beta * self.lastUpdateB[i] + (1 - self.beta) * self.nn.bUpdate[i] ** 2
            self.nn.weights[i] = self.nn.weights[i] - (self.lr * self.nn.bUpdate[i]) / (
                        np.sqrt(self.lastUpdateB[i]) + self.eps)


class ADAM:

    def __init__(self, nn, beta1=0.9, beta2=0.999, eps=1e-8, lr=0.001):
        self.nn = nn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.lastUpdateVW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateMW = [np.zeros_like(i) for i in nn.weights]

        self.lastUpdateVB = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateMB = [np.zeros_like(i) for i in nn.weights]

    def update(self):
        self.step += 1

        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateVW[i] = self.beta2 * self.lastUpdateVW[i] + (1 - self.beta2) * self.nn.wUpdate[i] ** 2
            self.lastUpdateMW[i] = self.beta1 * self.lastUpdateMW[i] + (1 - self.beta1) * self.nn.wUpdate[i]
            self.lastUpdateMhatW[i] = self.lastUpdateMW[i] / (1 - math.pow(self.beta1, self.step))
            self.lastUpdateVhatW[i] = self.lastUpdateVW[i] / (1 - math.pow(self.beta2, self.step))

            self.lastUpdateVB[i] = self.beta2 * self.lastUpdateVB[i] + (1 - self.beta2) * self.nn.bUpdate[i] ** 2
            self.lastUpdateMB[i] = self.beta1 * self.lastUpdateMB[i] + (1 - self.beta1) * self.nn.bUpdate[i]
            self.lastUpdateMhatB[i] = self.lastUpdateMB[i] / (1 - math.pow(self.beta1, self.step))
            self.lastUpdateVhatB[i] = self.lastUpdateVB[i] / (1 - math.pow(self.beta2, self.step))

            self.nn.weights[i] = self.nn.weights[i] - (self.lr * self.lastUpdateMhatW[i]) / (
            (np.sqrt(self.lastUpdateVhatW[i]) + self.eps))
            self.nn.bias[i] = self.nn.bias[i] - (self.lr * self.lastUpdateMhatB[i]) / (
            (np.sqrt(self.lastUpdateVhatB[i]) + self.eps))

