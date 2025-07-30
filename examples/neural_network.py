from microtorch.core.tensor import Tensor
from microtorch.core.linear import Linear
from microtorch.core.module import Module
from microtorch.core.functional import F

# Neural Network for MINST dataset
class NeuralNetwork(Module):

    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.l1 = Linear(128, input_dim)
        self.l2 = Linear(128, 128)
        self.l3 = Linear(output_dim, 128)
    
    def forward(self, x: Tensor):
        a1 = F.relu(self.l1(x))
        a2 = F.relu(self.l2(a1))
        a3 = F.softmax(self.l3(a2))
        return a3
    
