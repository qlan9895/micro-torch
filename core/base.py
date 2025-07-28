from ast import Tuple
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
	
	# Addition of two Tensors. C = A + B, then C will have C.op = "__add__"
	def __add__(self, other):
		result = Tensor(self.data + other.data,
			require_grad = self.require_grad or other.require_grad,
			op = "__add__")
		result._prev = (self, other)
		def _backward():
			if self.require_grad == True:
				if self.grad is None:
					self.grad = result.grad
				else:
					self.grad = self.grad + result.grad
			if other.require_grad == True:
				if other.grad is None:
					other.grad = result.grad
				else:
					other.grad = other.grad + result.grad
		result._backward = _backward
		return result

	# Matrix multiplcation
	def __matmul__(self, other):
		result = Tensor(self.data @ other.data, 
			require_grad = self.require_grad or other.require_grad,
			op = '__matmul__')
		result._prev = (self, other)
		def _backward():
			if self.require_grad == True:
				if self.grad is None:
					self.grad = result.grad @ other.data.T
				else:
					self.grad = self.grad + result.grad @ other.data.T
			if other.require_grad == True:
				if other.grad is None:
					other.grad = self.data.T @ result.grad
				else:
					other.grad = other.grad + self.data.T @ result.grad
		result._backward = _backward
		return result

	def __mul__(self, other):
		result = Tensor(self.data * other.data,
			require_grad = self.require_grad or other.require_grad,
			op = "__mul__")
		result._prev = (self, other)
		def _backward():
			if self.require_grad == True:
				if self.grad is None:
					self.grad = other.data * result.grad
				else:
					self.grad = self.grad + other.data * result.grad
			if other.require_grad == True:
				if other.grad is None:
					other.grad = self.data * result.grad
				else:
					other.grad = other.grad + self.data * result.grad
		result._backward = _backward
		return result

	def __sub__(self, other):
		result = Tensor(self.data - other.data,
			require_grad = self.require_grad or other.require_grad,
			op = "__sub__")
		result._prev = (self, other)
		def _backward():
			if self.require_grad == True:
				if self.grad is None:
					self.grad = result.grad
				else:
					self.grad += result.grad
			if other.require_grad == True:
				if other.grad is None:
					other.grad = - result.grad
				else:
					other.grad += - result.grad
		result._backward = _backward
		return result

	# Traverse the Tensor._prev from the current node. The data structure here is a DAG. Given an expression, we can generate a topology based on the expression.
	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)
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

	def __repr__(self):
		return f"Tensor(data={self.data}, require_grad={self.require_grad})"