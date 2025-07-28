from core.base import Tensor
from core.linear import Linear
from core.functional import F

# Neural Network for MINST dataset
class NeuralNetwork():

    def __init__(self, output_dim, input_dim):
        self.l1 = Linear(128, input_dim)
        self.l2 = Linear(128, 128)
        self.l3 = Linear(output_dim, 128)
    
    def forward(self, x: Tensor):
        self.a1 = F.relu(self.l1(x))
        self.a2 = F.relu(self.l2(self.a1))
        self.a3 = F.softmax(self.l3(self.a2))
        return self.a3
    
