import numpy as np
from .tensor import Tensor
from .functional import F
from .module import Module

# This is mainly for optimizer to grab parameters need to be updated. 

class Linear(Module):

	# initialize a Tensor with dimension (output_dim, input_dim)
	def __init__(self, output_dim, input_dim):
		super().__init__()
		self.W = Tensor(np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim), require_grad=True)
		self.b = Tensor(np.zeros((output_dim, 1)), require_grad=True)
	
	def forward(self, x):
		x = Tensor(x) if not isinstance(x, Tensor) else x
		result = self.W @ x + self.b
		return result

	def __call__(self, x):
		return self.forward(x)

	


		
	


		

















