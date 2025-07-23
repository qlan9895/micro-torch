import numpy as np

class LossF:

    @staticmethod
    # both y and y_pred should have output_dim * batch_size
    def MSE_loss(y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    @staticmethod
    # both y and y_pred should have output_dim * batch_size
    def cross_entropy(y, y_pred):
        return - np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=0))