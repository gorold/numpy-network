import numpynet as npn
import numpy as np

if __name__ == "__main__":
    data = npn.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
    target = npn.tensor([[0], [1], [0], [1]], autograd=True)

    w = list()
    w.append(npn.tensor(np.random.rand(2, 3), autograd=True))
    w.append(npn.tensor(np.random.rand(3, 1), autograd=True))

    optim = npn.optimizer.SGD(parameters=w, alpha=0.1)

    for i in range(100):
        pred = data.mm(w[0]).mm(w[1])
        loss = ((pred - target) * (pred - target)).sum(0)
        loss.backward(npn.tensor(np.ones_like(loss.data)))
        optim.step()

        print(loss)