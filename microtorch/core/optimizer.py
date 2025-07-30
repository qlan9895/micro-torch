from .tensor import Tensor

class Optimizer():

    def __init__(self, lr:float, weights):
        self.weights = weights
        self.lr = lr

    def update(self):
        for weight in self.weights:
            weight.data -= self.lr * weight.grad

    def zero_grad(self):
        for weight in self.weights:
            weight.grad = None