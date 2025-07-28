import numpy as np
from core.base import Tensor
from core.functional import F

class Linear:

	# initialize a Tensor with dimension (output_dim, input_dim)
	def __init__(self, output_dim, input_dim):
		self.W = Tensor(np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim), require_grad=True)
		self.b = Tensor(np.zeros((output_dim, 1)), require_grad=True)
	
	def forward(self, x):
		x = Tensor(x, require_grad=True) if not isinstance(x, Tensor) else x
		result = self.W @ x + self.b
		return result

	def __call__(self, x):
		return self.forward(x)

	


		
	


		

















