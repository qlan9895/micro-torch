import numpy as np

# Auto-grad implementation
class Tensor:

	def __init__(self, data, require_grad = False, grad = None, op=None):
		self.data = data
		self.require_grad = require_grad
		self.grad = grad # This should always store dL/d_data, if the Tensor is at the last of the list, self.grad = None
		self.op = op # How does one get this tensor, it determines how you back_propagate
		self._prev = ()
		self._backward = lambda: None
		self._reference_count = 0 # 
		self._reference_ready_count = 0
	
	# Addition of two Tensors. C = A + B, then C will have C.op = "__add__"
	def __add__(self, other):
		result = Tensor(self.data + other.data,
			require_grad = self.require_grad or other.require_grad,
			op = "__add__")
		result._prev = (self, other)
		self._reference_count = 1
		other._reference_count = 1
		def _backward():
			print("==========ENTER ADD BACKWARD===========")
			if self.require_grad == True and self._reference_ready_count == self._reference_count:
				if self.data.shape == result.data.shape:
					self.grad = self.grad + result.grad if self.grad is not None else result.grad
				else:
					# Handle broadcasting
					self.grad = self.grad + np.sum(result.grad, axis=1, keepdims=True) if self.grad is not None else np.sum(result.grad, axis=1, keepdims=True)
				for child in self._prev:
					child._reference_ready_count += 1 # Since the gradient computation is finished, the gradient is ready to use for downstream gradient computation
			if other.require_grad == True and other._reference_ready_count == other._reference_count:
				if other.data.shape == result.data.shape:
					other.grad = other.grad + result.grad if other.grad is not None else result.grad
				else:
					other.grad = other.grad + np.sum(result.grad, axis=1, keepdims=True) if other.grad is not None else np.sum(result.grad, axis=1, keepdims=True)
				for child in other._prev:
					child._reference_ready_count += 1
		result._backward = _backward
		return result

	# Matrix multiplcation
	def __matmul__(self, other):
		result = Tensor(self.data @ other.data, 
			require_grad = self.require_grad or other.require_grad,
			op = '__matmul__')
		result._prev = (self, other)
		self._reference_count = 1
		other._reference_count = 1
		def _backward():
			if self.require_grad == True and self._reference_count == self._reference_ready_count:
				if self.grad is None:
					self.grad = result.grad @ other.data.T
				else:
					self.grad = self.grad + result.grad @ other.data.T
				for child in self._prev:
					child._reference_ready_count += 1
			if other.require_grad == True and other._reference_count == other._reference_ready_count:
				if other.grad is None:
					other.grad = self.data.T @ result.grad
				else:
					other.grad = other.grad + self.data.T @ result.grad
				for child in other._prev:
					child._reference_ready_count += 1
		result._backward = _backward
		return result

	# Elementwise matrix multiplication
	def __mul__(self, other):
		result = Tensor(self.data * other.data,
			require_grad = self.require_grad or other.require_grad,
			op = "__mul__")
		result._prev = (self, other)
		self._reference_count = 1
		other._reference_count = 1
		def _backward():
			if self.require_grad == True and self._reference_count == self._reference_ready_count:
				# Handle broadcasting
				if self.data.shape == other.data.shape:
					self.grad = self.grad + other.data * result.grad if self.grad is not None else other.data * result.grad
				else:
					self.grad = self.grad + np.sum(other.data * result.grad, axis=1, keepdims=True) if self.grad is not None else np.sum(other.data * result.grad, axis=1, keepdims=True)
				for child in self._prev:
					child._reference_ready_count += 1
			if other.require_grad == True and other._reference_count == other._reference_ready_count:
				if other.data.shape == self.data.shape:
					other.grad = other.grad + self.data * result.grad if other.grad is not None else self.data * result.grad
				else:
					other.grad = other.grad + np.sum(self.data * result.grad, axis=1, keepdims=True) if other.grad is not None else np.sum(self.data * result.grad, axis=1, keepdims=True)
				for child in other._prev:
					child._reference_ready_count += 1
		result._backward = _backward
		return result

	#Subtraction
	def __sub__(self, other):
		result = Tensor(self.data - other.data,
			require_grad = self.require_grad or other.require_grad,
			op = "__sub__")
		result._prev = (self, other)
		self._reference_count = 1
		other._reference_count = 1
		def _backward():
			if self.require_grad == True and self._reference_count == self._reference_ready_count:
				if self.grad is None:
					self.grad = result.grad
				else:
					self.grad += result.grad
				for child in self._prev:
					child._reference_ready_count += 1
			if other.require_grad == True and other._reference_count == other._reference_ready_count:
				if other.grad is None:
					other.grad = - result.grad
				else:
					other.grad += - result.grad
				for child in other._prev:
					child._reference_ready_count += 1
		result._backward = _backward
		return result

	# Traverse the Tensor._prev from the current node. The data structure here is a DAG. Given an expression, we can generate a topology based on the expression.
	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)
		for child in self._prev:
			child._reference_ready_count += 1
		visited = set()
		traversed_order = []
		# for C = A * B + A, we have visited = {A, B, C} and topo = [A, B, A * B, C], this is a resemblance to topology.
		def build_order(tensor):
			if tensor not in visited:
				visited.add(tensor)
				if tensor._prev is not None:
					for child in tensor._prev:
						build_order(child)
				traversed_order.append(tensor)
				
		build_order(self)
		for t in reversed(traversed_order):
			t._backward()
			breakpoint()

	def __repr__(self):
		return f"Tensor(data={self.data}, require_grad={self.require_grad})"
	
	# Used for Module class to track whether a Tensor is needed to get updated. If self._leaf is True then
	# Module add tracks them
	def is_leaf(self):
		self._leaf = True if len(self._prev) == 0 else False
		return self._leaf

	# split matrix with respect to columns
	def split(self, idx):
		a, b = self.data[:,:idx], self.data[:,idx:]
		result_a, result_b = Tensor(a, require_grad=self.require_grad, op="split"), Tensor(b, require_grad=self.require_grad, op="split")
		result_a._prev, result_b._prev = (self, ), (self, )
		self._reference_count = 2 # As it output two Tensors
		def _backward():
			print("=======ENTERING BACKWARD========")
			if self.require_grad == True and self._reference_count == self._reference_ready_count:
				if self.grad is None:
					self.grad = np.concatenate((result_a.grad, result_b.grad), axis=1)
				else:
					self.grad += np.concatenate((result_a.grad, result_b.grad), axis=1)
				breakpoint()
				for child in self._prev:
					child._reference_ready_count += 1
		result_a._backward = _backward
		result_b._backward = _backward
		return result_a, result_b

	# slice matrix with respect to rows
	def slice(self, idx):
		a, b = self.data[:idx,:], self.data[idx:,:]
		result_a, result_b = Tensor(a, require_grad=self.require_grad, op="slice"), Tensor(b, require_grad=self.require_grad, op="slice")
		result_a._prev, result_b._prev = (self, ), (self, )
		self._reference_count = 2
		def _backward():
			if self.require_grad == True and self._reference_count == self._reference_ready_count:
				if self.grad is None:
					self.grad = np.concatenate((result_a.grad, result_b.grad), axis=0)
				else:
					self.grad += np.concatenate((result_a.grad, result_b.grad), axis=0)
				for child in self._prev:
					child._reference_ready_count += 1
		result_a._backward = _backward
		result_b._backward = _backward
		return result_a, result_b
