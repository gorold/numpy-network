import numpy as np

class tensor():
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

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

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
                    self.parents[0].backward(grad, self)
                    self.parents[1].backward(grad, self)
