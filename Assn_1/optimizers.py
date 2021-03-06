import numpy as np
import math

class SGD:

    def __init__(self, nn, lr=0.001 , wd=0 ):
        """
        In SGD every update is after a single data point. The loss may or maynot decrease.
        :param nn:neural network object
        :param lr: learning rate (eta)
        :param wd: weight decay used for regularisation
        """
        self.nn = nn
        self.lr = lr
        self.wd = wd
       


    def update(self):
        """
        :return: The weights and bias are updated according to the type of optimiser
        """
        for i in range(self.nn.numOfLayers - 1):
            self.nn.weights[i] = self.nn.weights[i] - self.lr * self.nn.wUpdate[i] - self.lr * self.wd *self.nn.weights[i]
            self.nn.bias[i] = self.nn.bias[i] - self.lr * self.nn.bUpdate[i] - self.lr * self.wd *self.nn.bias[i]

class MOMEMTUM:

    def __init__(self, nn, gamma=0.9, lr=0.001 , wd=0):
        """
        The history is also taken into account along with the current gradient
        :param nn:neural network object
        :param gamma: const for scaling the history term
        :param lr: learning rate (eta)
        :param wd: weight decay used for regularisation
        """
        self.nn = nn
        self.lr = lr
        self.gamma = gamma
        self.wd = wd

        self.lastUpdateW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB = [np.zeros_like(i) for i in nn.bias]

    def update(self):
        """
        :return: The weights and bias are updated according to the type of optimiser
        """
        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateW[i] = (self.gamma * self.lastUpdateW[i]) + (self.lr * self.nn.wUpdate[i]) + self.lr * self.wd *self.nn.weights[i]
            self.nn.weights[i] = self.nn.weights[i] - self.lastUpdateW[i]

            self.lastUpdateB[i] = (self.gamma * self.lastUpdateB[i]) + (self.lr * self.nn.bUpdate[i]) + self.lr * self.wd *self.nn.bias[i]
            self.nn.bias[i] = self.nn.bias[i] - self.lastUpdateB[i]


class NESTEROV:

    def __init__(self, nn, gamma=0.9, lr=0.001 , wd=0):
        """
        Momemtum leads to many oscillations so the lookahead point is taken into consideration
        :param nn: neural network object
        :param gamma: const for scaling the history term
        :param lr: learning rate (eta)
        :param wd: weight decay used for regularisation
        """
        self.nn = nn
        self.lr = lr
        self.gamma = gamma
        self.wd = wd

        self.lastUpdateW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB = [np.zeros_like(i) for i in nn.bias]

    def partialUpdate(self):
        """
        This function is used to calculate the gradients at the look ahead points.
        :return: gradients at look ahead point
        """
        self.originalw = self.nn.weights
        self.originalb = self.nn.bias

        wahead = []
        for i in range(self.nn.numOfLayers-1):
            wahead.append(self.nn.weights[i] - self.gamma * self.lastUpdateW[i])
        self.nn.weights = wahead

        bahead = []
        for i in range(self.nn.numOfLayers-1):
            bahead.append(self.nn.bias[i] - self.gamma * self.lastUpdateB[i])
        self.nn.bias = bahead


    def update(self):
        """
        :return: The weights and bias are updated according to the type of optimiser
        """
        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateW[i] = (self.gamma * self.lastUpdateW[i]) + (self.lr * self.nn.wUpdate[i]) + self.lr * self.wd *self.nn.weights[i]
            self.nn.weights[i] = self.originalw[i] - self.lastUpdateW[i]

            self.lastUpdateB[i] = (self.gamma * self.lastUpdateB[i]) + (self.lr * self.nn.bUpdate[i]) + self.lr * self.wd *self.nn.bias[i]
            self.nn.bias[i] = self.originalb[i] - self.lastUpdateB[i]


class RMSPROP:

    def __init__(self, nn, beta=0.9, eps=1e-8, lr=0.001 , wd=0):
        """
        :param nn: neural network object
        :param beta: constant
        :param eps: epochs
        :param lr: learning rate(eta)
        :param wd: weight decay
        """
        self.nn = nn
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.wd = wd

        self.lastUpdateW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB = [np.zeros_like(i) for i in nn.bias]

    def update(self):
        """
        :return: The weights and bias are updated according to the type of optimiser
        """
        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateW[i] = self.beta * self.lastUpdateW[i] + (1 - self.beta) * self.nn.wUpdate[i] ** 2
            self.nn.weights[i] = self.nn.weights[i] - self.lr * self.wd *self.nn.weights[i]- (self.lr * self.nn.wUpdate[i]) / (
                        np.sqrt(self.lastUpdateW[i] + self.eps))

            self.lastUpdateB[i] = self.beta * self.lastUpdateB[i] + (1 - self.beta) * self.nn.bUpdate[i] ** 2
            self.nn.bias[i] = self.nn.bias[i] - self.lr * self.wd *self.nn.bias[i]-(self.lr * self.nn.bUpdate[i]) / (
                        np.sqrt(self.lastUpdateB[i] + self.eps))


class ADAM:

    def __init__(self, nn, beta1=0.9, beta2=0.999, eps=1e-8, lr=0.001 , wd=0):
        """
        :param nn: neuarl network object
        :param beta1: constant
        :param beta2: constant
        :param eps: num of epochs
        :param lr: learning rate
        :param wd: weight decay
        """
        self.nn = nn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.step = 0
        self.lastUpdateVW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateMW = [np.zeros_like(i) for i in nn.weights]

        self.lastUpdateVB = [np.zeros_like(i) for i in nn.bias]
        self.lastUpdateMB = [np.zeros_like(i) for i in nn.bias]

    def update(self):
        """
        :return: The weights and bias are updated according to the type of optimiser
        """
        self.step += 1

        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateVW[i] = self.beta2 * self.lastUpdateVW[i] + (1 - self.beta2) * self.nn.wUpdate[i] ** 2
            self.lastUpdateMW[i] = self.beta1 * self.lastUpdateMW[i] + (1 - self.beta1) * self.nn.wUpdate[i]
            self.lastUpdateMhatW = self.lastUpdateMW[i] / (1 - math.pow(self.beta1, self.step))
            self.lastUpdateVhatW = self.lastUpdateVW[i] / (1 - math.pow(self.beta2, self.step))

            self.lastUpdateVB[i] = self.beta2 * self.lastUpdateVB[i] + (1 - self.beta2) * self.nn.bUpdate[i] ** 2
            self.lastUpdateMB[i] = self.beta1 * self.lastUpdateMB[i] + (1 - self.beta1) * self.nn.bUpdate[i]
            self.lastUpdateMhatB = self.lastUpdateMB[i] / (1 - math.pow(self.beta1, self.step))
            self.lastUpdateVhatB = self.lastUpdateVB[i] / (1 - math.pow(self.beta2, self.step))

            self.nn.weights[i] = self.nn.weights[i] - self.lr * self.wd *self.nn.weights[i]-(self.lr * self.lastUpdateMhatW) / (
            (np.sqrt(self.lastUpdateVhatW + self.eps)))
            self.nn.bias[i] = self.nn.bias[i] - self.lr * self.wd *self.nn.bias[i]- (self.lr * self.lastUpdateMhatB) / (
            (np.sqrt(self.lastUpdateVhatB + self.eps)))


class NADAM:

    def __init__(self, nn, beta1=0.9, beta2=0.999, eps=1e-8, lr=0.001, gamma=0.9,wd=0):
        """
        :param nn: neural network object
        :param beta1: constant
        :param beta2: constant
        :param eps: number of epochs
        :param lr: learning rate
        :param gamma: momemtum constant
        :param wd: weight decay
        """
        self.nn = nn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.gamma = gamma
        self.wd = wd
        self.lastUpdateVW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateMW = [np.zeros_like(i) for i in nn.weights]

        self.lastUpdateW = [np.zeros_like(i) for i in nn.weights]
        self.lastUpdateB = [np.zeros_like(i) for i in nn.bias]

        self.lastUpdateVB = [np.zeros_like(i) for i in nn.bias]
        self.lastUpdateMB = [np.zeros_like(i) for i in nn.bias]

    def partialUpdate(self):
        """
        :return: calculates the gradient at a look ahead point
        """

        self.originalw = self.nn.weights
        self.originalb = self.nn.bias

        wahead = []
        for i in range(self.nn.numOfLayers - 1):
            wahead.append(self.nn.weights[i] - self.gamma * self.lastUpdateW[i])
        self.nn.weights = wahead

        bahead = []
        for i in range(self.nn.numOfLayers - 1):
            bahead.append(self.nn.bias[i] - self.gamma * self.lastUpdateB[i])
        self.nn.bias = bahead

    def update(self):
        """
        :return: The weights and bias are updated according to the type of optimiser
        """
        self.step += 1

        for i in range(self.nn.numOfLayers - 1):
            self.lastUpdateVW[i] = self.beta2 * self.lastUpdateVW[i] + (1 - self.beta2) * self.nn.wUpdate[i] ** 2
            self.lastUpdateMW[i] = self.beta1 * self.lastUpdateMW[i] + (1 - self.beta1) * self.nn.wUpdate[i]
            self.lastUpdateMhatW = self.lastUpdateMW[i] / (1 - math.pow(self.beta1, self.step))
            self.lastUpdateVhatW = self.lastUpdateVW[i] / (1 - math.pow(self.beta2, self.step))

            self.lastUpdateVB[i] = self.beta2 * self.lastUpdateVB[i] + (1 - self.beta2) * self.nn.bUpdate[i] ** 2
            self.lastUpdateMB[i] = self.beta1 * self.lastUpdateMB[i] + (1 - self.beta1) * self.nn.bUpdate[i]
            self.lastUpdateMhatB = self.lastUpdateMB[i] / (1 - math.pow(self.beta1, self.step))
            self.lastUpdateVhatB = self.lastUpdateVB[i] / (1 - math.pow(self.beta2, self.step))

            self.lastUpdateW[i] = (self.gamma * self.lastUpdateW[i]) + self.lr * self.wd *self.nn.weights[i]+(self.lr * self.lastUpdateMhatW) / (
                (np.sqrt(self.lastUpdateVhatW + self.eps)))

            self.lastUpdateB[i] = (self.gamma * self.lastUpdateB[i]) + self.lr * self.wd *self.nn.bias[i]+(self.lr * self.lastUpdateMhatB) / (

                (np.sqrt(self.lastUpdateVhatB + self.eps)))

            self.nn.weights[i] = self.originalw[i] - self.lastUpdateW[i]
            self.nn.bias[i] = self.originalb[i] - self.lastUpdateB[i]

optDict = {
    'SGD':SGD,
    'NESTEROV':NESTEROV,
    'MOMEMTUM':MOMEMTUM,
    'ADAM':ADAM,
    'NADAM':NADAM,
    'RMSPROP':RMSPROP
}
