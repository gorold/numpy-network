import numpynet as npn
import numpy as np

if __name__ == "__main__":
    a = npn.tensor([1, 2, 3, 4, 5], autograd=True, id='a')
    b = npn.tensor([2, 2, 2, 2, 2], autograd=True, id='b')
    c = npn.tensor([5, 4, 3, 2, 1], autograd=True, id='c')
    d = a + b
    e = b + c
    f = d + e
    f.backward(npn.tensor([1, 1, 1, 1, 1]))
    print(b.grad.data)