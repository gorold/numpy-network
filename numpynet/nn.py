import numpy as np
from numpynet.tensor import tensor

class Layer(object):
    
    def __init__(self, module=None):
        self.parameters = list()
        if module is not None:
            module.add_layer(self)

    def __call__(self, x):
        if type(x) is not tuple:
            return self.forward(x)
        else: 
            if len(x) == 2:
                return self.forward(x[0], x[1])

    def get_parameters(self):
        return self.parameters

    def forward(self, x):
        raise Exception("Please implement forward method.")

    def forward(self, y_hat, y):
        raise Exception("Please implement forward method")

class Module(object):
    def __init__(self):
        self.layers = list()

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params

class Linear(Layer):

    def __init__(self, n_inputs, n_outputs, module):
        super().__init__(module)
        # Xavier initialization (good for tanh)
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs+n_outputs)) 
        self.weight = tensor(W, autograd=True)
        self.bias = tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))

class MSELoss(Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)

