import numpy as np

class NeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_size):
        self.param = {}
        self.param["W1"] = np.random.rand(hidden_size, input_dim)
        self.param["W2"] = np.random.rand(hidden_size, hidden_size)
        self.param["W3"] = np.random.rand(output_dim, hidden_size)
        self.pre_forward_pass = []
        self.post_forward_pass = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # x is the input data of the dimension m, we use sigmoid function for activation
    def forward(self, x):

        self.pre_forward_pass = []
        self.post_forward_pass = []

        y = np.dot(self.param["W1"], x)
        self.pre_forward_pass.append(y)
        y = self.sigmoid(y)
        self.post_forward_pass.append(y)

        y = np.dot(self.param["W2"], y)
        self.pre_forward_pass.append(y)
        y = self.sigmoid(y)
        self.post_forward_pass.append(y)

        y = np.dot(self.param["W3"], y)
        self.pre_forward_pass.append(y)
        y_pred = self.sigmoid(y)
        self.post_forward_pass.append(y_pred)
        return y_pred

    # Both y and y_pred should have dimension: output_dim * training_samples
    def MSE_loss(self, y):
        y_pred = self.post_forward_pass[-1]
        loss = 0.5 * np.mean((y - y_pred) ** 2)
        return loss
    
    def  cross_entropy_loss(self, y):
        y_pred = 
    # Back propagation implementation
    # We first reverse the post_forward_pass list and then we compute the gradient for each layer using chain rule
    def backward(self, y, x):
        grad_param = {}

        y_pred = self.post_forward_pass[-1]  # shape: output_dim x num_samples
        dL_da3 = y_pred - y  # Derivative of MSE loss

        # Layer 3 (output layer)
        a2 = self.post_forward_pass[1] # shape: hidden_size x num_sampels
        da3_dz3 = y_pred * (1 - y_pred) # shape: output_dim x num_samples
        dz3_dw3 = a2
        dL_dz3 = dL_da3 * da3_dz3 #shape output_dim x num_samples
        grad_param["W3"] = np.dot(dL_dz3, dz3_dw3.T) #shape: output_dim x hidden_size

        # Layer 2 (hidden layer)
        a1 = self.post_forward_pass[0]
        da2_dz2 = a2 * (1 - a2)
        dL_da2 = np.dot(self.param["W3"].T, dL_dz3) # dL_dz3 * dz3_da3, dz3_da3 is just W3
        dL_dz2 = dL_da2 * da2_dz2
        grad_param["W2"] = np.dot(dL_dz2, a1.T)

        # Layer 1 (input layer)
        da1_dz1 = a1 * (1 - a1)
        dL_dz1 = np.dot(self.param["W2"].T, dL_dz2) * da1_dz1
        grad_param["W1"] = np.dot(dL_dz1, x.T)

        return grad_param
    
    



            
            
