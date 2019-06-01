import numpy as np

class Optimizer(object):
    
    def __init__(self, parameters):
        self.parameters = parameters

    def zero(self):
        for p in self.parameters:
            self.single_zero(p)

    def single_zero(self, p):
        p.grad.data *= 0

class SGD(Optimizer):
    def __init__(self, parameters, alpha=0.1):
        super().__init__(parameters)
        self.alpha = alpha

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha
            if zero: self.single_zero(p)

class Adam(Optimizer):

    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.use_locking = use_locking
        self.m = self.initMV()
        self.v = self.initMV()

    def initMV(self):
        new_param_list = list()
        for p in self.parameters:
            new_param = np.zeros_like(p)
            new_param_list.append(new_param)

        return new_param_list

    def step(self, zero=True):
        for i in range(len(self.parameters)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.parameters[i].data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (self.parameters[i].data * self.parameters[i].data)
            self.parameters[i].data -= (self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon))
            if zero: self.single_zero(self.parameters[i])
