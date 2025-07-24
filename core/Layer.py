import numpy as np

class F:

	@staticmethod
	def sigmoid(v):
		return 1 / 1 + np.exp(-v)

	@staticmethod
	def relu(v):
		return np.maximum(v, 0)
	
	@staticmethod
	def softmax(v):
		v_max = np.max(v, axis=0, keepdims=True)       # shape: (output_dim, batch_size)
		e_v = np.exp(v - v_max)                         # shift for stability
		return e_v / np.sum(e_v, axis=0, keepdims=True)

	@staticmethod
	def relu_grad(v):
		return (v > 0).astype(float)
	
	@staticmethod
	def tanh(v):
		return np.tanh(v)
	
	@staticmethod
	def tanh_grad(v):
		return 1 - np.tanh(v) ** 2

class Layer:

	# initialize a matrix with dimension (output_dim, input_dim)
	def __init__(self, output_dim, input_dim, activation=None, W_grad=None, b_grad=None):
		self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
		self.b = np.zeros((output_dim, 1))
		self.activation = activation
		self.W_grad = W_grad
		self.b_grad = b_grad
	
	def forward(self, x):
		self.input = x
		self.out = self.W @ x + self.b
		return self.out

	def __call__(self, x):
		return self.forward(x)

	


		
	


		

















