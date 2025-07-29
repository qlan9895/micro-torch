from core.tensor import Tensor
from core.linear import Linear
from core.functional import F
from core.module import Module

class NeuralNetwork(Module):

    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.l1 = Linear(2, input_dim)
        self.l2 = Linear(output_dim, 2)

    def forward(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        a1 = F.relu(self.l1(x))
        a2 = F.softmax(self.l2(a1))
        return a2

    def __call__(self,x):
        return self.forward(x)

nn = NeuralNetwork(2, 2)

def test_module_params():
    assert nn._params == {}
    print(nn._params)

def test_module_get_modules():
    assert nn._modules == {'l1': nn.l1, 'l2': nn.l2}
    print(nn._modules)

def test_module_get_params():
    print(nn.params())