import numpy as np

class tensor(object):
    _id_counter = 0

    def __init__(self, data, autograd=False, op=None, id=None, parents=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.op = op
        self.id = self.next_id if id is None else id
        self.parents = parents
        self._generate_children_for_parents()
        self.grad = None


    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __add__(self, other):
        if self.autograd and other.autograd:
            return tensor(self.data + other.data, autograd=True, op="add", parents=[self, other])
        return tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return tensor(self.data * -1, autograd=True, parents=[self], op="neg")
        return tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return tensor(self.data - other.data, autograd=True,op="sub", parents=[self, other])
        return tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return tensor(self.data * other.data, autograd=True, op="mul", parents=[self, other])
        return tensor(self.data * other.data)
    
    def sum(self, dim):
        if self.autograd:
            return tensor(self.data.sum(dim), autograd=True, parents=[self], op="sum_" + str(dim))
        return tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return tensor(new_data, autograd=True, parents=[self], op="expand_" + str(dim))
        return tensor(new_data)

    def transpose(self):
        if self.autograd:
            return tensor(self.data.transpose(), autograd=True, parents=[self], op="transpose")
        return tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return tensor(self.data.dot(x.data), autograd=True, parents=[self, x], op="mm")
        return tensor(self.data.dot(x.data))

    def _generate_children_for_parents(self):
        self.children = {}
        if self.parents is not None:
            for p in self.parents:
                p.children[self.id] = p.children.get(self.id, 0) + 1

    def _check_and_decrement_origin(self, child_origin):
        if child_origin is not None:
                if self.children[child_origin.id] == 0:
                    raise Exception("Backprop limit reached")
                else:
                    self.children[child_origin.id] -= 1

    def _update_grad(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def _all_children_grads_accounted_for(self):
        for k, v in self.children.items():
            if v != 0:
                return False
        return True

    @property
    def next_id(self):
        tensor._id_counter += 1
        return tensor._id_counter

    def backward(self, grad=None, child_origin=None):
        if self.autograd:
            self._check_and_decrement_origin(child_origin)
            self._update_grad(grad)
            if self.parents is not None and (self._all_children_grads_accounted_for() or child_origin is None):
                if self.op == "add":
                    self.parents[0].backward(self.grad, self)
                    self.parents[1].backward(self.grad, self)
                if self.op == "neg":
                    self.parents[0].backward(self.grad.__neg__())
                if self.op == "sub":
                    self.parents[0].backward(self.grad, self)
                    self.parents[1].backward(self.grad.__neg__(), self)
                if self.op == "mul":
                    temp = self.grad * self.parents[1]
                    self.parents[0].backward(temp, self)
                    temp = self.grad * self.parents[0]
                    self.parents[1].backward(temp, self)
                if "sum" in self.op:
                    dim = int(self.op.split("_")[1])
                    ds = self.parents[0].data.shape[dim]
                    self.parents[0].backward(self.grad.expand(dim, ds))
                if "expand" in self.op:
                    dim = int(self.op.split("_")[1])
                    self.parents[0].backward(self.grad.sum(dim))
                if self.op == "transpose":
                    self.parents[0].backward(self.grad.transpose())
                if self.op == "mm":
                    act = self.parents[0]
                    weights = self.parents[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)