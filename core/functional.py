from core.base import Tensor
import numpy as np

class F:

    @staticmethod
    def sigmoid(v: Tensor) -> Tensor:
        data = 1 / (1 + np.exp(-v.data))
        result = Tensor(data, require_grad = v.require_grad, op = 'sigmoid')
        result._prev = (v, )
        def _backward():
            if result.require_grad == True:
                if v.grad == None:
                    v.grad = result.grad * data * (1 - data)
                else:
                    v.grad += result.grad * data * (1 - data)
        result._backward = _backward
        return result

    @staticmethod
    def relu(v: Tensor) -> Tensor:
        data = np.maximum(v.data, 0)
        result = Tensor(data, require_grad = v.require_grad, op = 'relu')
        result._prev = (v, )
        def _backward():
            if result.require_grad == True:
                if v.grad == None:
                    v.grad = result.grad * (v.data > 0).astype(float)
                else:
                    v.grad += result.grad * (v.data > 0).astype(float)
        result._backward = _backward
        return result

    @staticmethod
    def softmax(v: Tensor) -> Tensor:
        v_max = np.max(v.data, axis=0, keepdims=True)       # shape: (output_dim, batch_size)
        e_v = np.exp(v.data - v_max)                         # shift for stability
        data =  e_v / np.sum(e_v, axis=0, keepdims=True)
        result = Tensor(data, require_grad = v.require_grad, op = 'softmax')
        result._prev = (v, )
        def _backward():
            # dL/dS has to be a matrice that has the same size as v. Since we only need to store dL/dv in v.grad, we don't compute all jacobian matrices of dS/dv
            # We get each column of matrix of dL/dS and matmul with the corresponding Jacobians
            if result.require_grad == True:
                if v.grad == None:
                    grad = []
                    for i in range(result.data.shape[1]):
                        vec = result.data[:,i:i+1]
                        small_grad = np.diag(vec.squeeze()) - vec @ vec.T
                        grad.append(small_grad @ result.grad[:,i:i+1])
                    v.grad = np.concatenate(grad, axis=1)
                else:
                    grad = []
                    for i in range(result.data.shape[1]):
                        vec = result.data[:,i:i+1]
                        small_grad = np.diag(vec.squeeze()) - vec @ vec.T
                        grad.append(small_grad @ result.grad[:,i:i+1])
                    v.grad += np.concatenate(grad, axis=1)
        result._backward = _backward     
        return result
    
    @staticmethod
    def tanh(v: Tensor) -> Tensor:
        data = np.tanh(v.data)
        result = Tensor(data, require_grad = v.require_grad, op = 'tanh')
        result._prev = (v, )
        def _backward():
            if result.require_grad == True:
                if v.grad == None:
                    v.grad = result.grad * (1 - result.data ** 2)
                else:
                    v.grad += result.grad * (1 - result.data ** 2)
        result._backward = _backward
        return result
    
