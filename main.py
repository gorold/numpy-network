import numpynet as npn
from numpynet import optimizer
from numpynet import nn
import numpy as np

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, self)
        self.fc2 = nn.Linear(3, 1, self)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    data = npn.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
    target = npn.tensor([[0], [1], [0], [1]], autograd=True)

    # w = list()
    # w.append(npn.tensor(np.random.rand(2, 3) * np.sqrt(2/(2+3)), autograd=True))
    # w.append(npn.tensor(np.random.rand(3, 1) * np.sqrt(2/(3+1)), autograd=True))

    model = MyModel()
    criterion = nn.MSELoss()

    optim = npn.optimizer.Adam(parameters=model.get_parameters())

    for i in range(1000):
        pred = model.forward(data)
        loss = criterion((pred, target))
        loss.backward(npn.tensor(np.ones_like(loss.data)))
        optim.step()

        print(loss)