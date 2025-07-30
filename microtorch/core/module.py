from .tensor import Tensor

class Module:

	def __init__(self):
		self._params = {}
		self._modules = {}

	def __setattr__(self, name, value):
		if isinstance(value, Tensor):
			self._params[name] = value
			object.__setattr__(self, name, value)
		elif isinstance(value, Module):
			self._modules[name] = value
			object.__setattr__(self, name, value)
		else:
			object.__setattr__(self, name, value)

	# Output a list of Tensors that need to be updated, we do this by traversing its submodules recursively
	def params(self):
		params = []
		#iterate over current self._params
		for tensor in self._params.values():
			if tensor.is_leaf() == True:
				params.extend([tensor])
		for submodule in self._modules.values():
			params.extend(submodule.params())

		return params