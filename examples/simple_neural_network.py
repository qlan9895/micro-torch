from core.linear import Linear
from core.functional import F
from core.base import Tensor
import numpy as np

class NeuralNetwork():

    def __init__(self, output_dim, input_dim, hidden_size):
        self.l1 = Linear(hidden_size, input_dim)
        self.l2 = Linear(hidden_size, hidden_size)
        self.l3 = Linear(output_dim, hidden_size)
    
    # Implementation of forward pass: post_* should represent the activation value after an Linear layer
    # dim(x) = input_dim * batch_size
    def forward(self, x): 
        a1 = F.relu(self.l1(x))
        a2 = F.relu(self.l2(a1))
        a3 = F.softmax(self.l3(a2))
        return a3
    
    # Implementation of backward pass: 

    def loss(self, y, y_pred):
        return LossF.cross_entropy(y, y_pred)

    def backward(self, y, y_pred):
        # dL_dl3 = (dL / a3 * da3 / dl3)
        dL_dl3 = y_pred - y # the gradient between cross_entropy loss and value before softmax. Dimension: output_dim * batch_size
        dl3_dW3 = self.l3.input.T # the gradient between linear and weight, dim(dl3_dW3) = batch_size * hidden_size
        dL_dW3 = dL_dl3 @ dl3_dW3 # output_dim * batch_size
        dL_db3 = np.sum(dL_dl3, axis=1, keepdims=True)
        self.l3.W_grad = dL_dW3
        self.l3.b_grad = dL_db3

        # dL_da2 = dL_dl3 * dl3_da2 
        dl3_da2 = self.l3.W.T # dimension: hidden_size * output_dim. Note that it's linear
        da2_dl2 = F.relu_grad(self.a2) # dimension = dim(self.l2.out) = hidden_size * batch_size
        dL_dl2 =  (dl3_da2 @ dL_dl3) * da2_dl2 # dimension = (hid, out) @ (out, batch_size) * (hid, batch_size)
        dl2_dW2 = self.l2.input.T # batch_size * hidden_size
        dL_dW2 = dL_dl2 @ dl2_dW2
        dL_db2 = np.sum(dL_dl2, axis=1, keepdims=True)
        self.l2.W_grad = dL_dW2
        self.l2.b_grad = dL_db2

        dl2_da1 = self.l2.W.T # dimension: hidden_size * hidden_size
        da1_dl1 = F.relu_grad(self.a1)
        dL_dl1 = (dl2_da1 @ dL_dl2) * da1_dl1
        dl1_dW1 = self.l1.input.T
        dL_dW1 = dL_dl1 @ dl1_dW1
        dL_db1 = np.sum(dL_dl1, axis=1, keepdims=True)
        self.l1.W_grad = dL_dW1
        self.l1.b_grad = dL_db1