from .tensor import Tensor
import numpy as np

class F:

    # Activation functions
    @staticmethod
    def sigmoid(v: Tensor) -> Tensor:
        data = 1 / (1 + np.exp(-v.data))
        result = Tensor(data, require_grad = v.require_grad, op = 'sigmoid')
        result._prev = (v, )
        def _backward():
            if result.require_grad == True:
                if v.grad is None:
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
                if v.grad is None:
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
                if v.grad is None:
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
                if v.grad is None:
                    v.grad = result.grad * (1 - result.data ** 2)
                else:
                    v.grad += result.grad * (1 - result.data ** 2)
        result._backward = _backward
        return result
    
    # Loss functions
    @staticmethod
    def mse_loss(v: Tensor, v_pred: Tensor) -> Tensor:
        # sum all inner product with respect to columns
        data_list = []
        for i in range(v.data.shape[1]):
            inner = np.linalg.norm(v_pred.data[:,i:i+1] - v.data[:,i:i+1]) ** 2
            data_list.append(inner)
        data = np.mean(data_list)
        result = Tensor(data, require_grad = v.require_grad or v_pred.require_grad, op='mse_loss')
        result._prev = (v_pred, v)
        def _backward():
            if v_pred.require_grad == True:
                if v_pred.grad is None:
                    v_pred.grad = (2/v_pred.data.shape[1]) * (v_pred.data - v.data) * result.grad
                else:
                    v_pred.grad += (2/v_pred.data.shape[1]) * (v_pred.data - v.data) * result.grad
            if v.require_grad == True:
                if v.grad is None:
                    v.grad = (2/v_pred.data.shape[1]) * (v.data - v_pred.data) * result.grad
                else:
                    v.grad += (2/v_pred.data.shape[1]) * (v.data - v_pred.data) * result.grad
        result._backward = _backward
        return result

    @staticmethod
    def cross_entropy(v: Tensor, v_pred: Tensor) -> Tensor:
        data_list = []
        for i in range(v.data.shape[1]):
            inner = np.dot(v.data[:,i:i+1].T, np.log(v_pred.data[:,i:i+1]))
            data_list.append(inner)
        data = - np.sum(data_list)
        result = Tensor(data, require_grad = v.require_grad or v_pred.require_grad, op='cross_entropy')
        result._prev = (v_pred, v)
        def _backward():
            if v_pred.require_grad == True:
                if v_pred.op == 'softmax':
                    v_pred._backward = lambda: None # Since by calling loss.backward(), softmax is already in the DAG, so we need to short-circuit it. 
                    logits = v_pred._prev[0] # Returns a Tensor, this Tensor was fed into softmax
                    if logits.grad is None:
                        logits.grad = (v_pred.data - v.data) * result.grad
                    else:
                        logits.grad += (v_pred.data - v.data) * result.grad
                     
                ### TO DO: implement the backward step when v_pred.op is not 'softmax'
                ### Code Here
        result._backward = _backward
        return result

                        



        
