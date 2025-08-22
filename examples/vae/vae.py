from microtorch.core.module import Module
from microtorch.core.functional import F
from microtorch.core.linear import Linear
from microtorch.core.tensor import Tensor
import numpy as np

class NeuralNetwork(Module):

    '''
    if latent space have dimension d, the output dimension should have dimension 2d:
        \mu(x) should have dimension d
        \sigma(x) should have dimension d
    '''
    def __init__(self, output_dim: int, input_dim: int):
        super().__init__()
        self.w1 = Linear(128, input_dim)
        self.w2 = Linear(output_dim, 128)
    
    def forward(self, x:Tensor):
        a1 = F.sigmoid(self.w1(x))
        a2 = F.sigmoid(self.w2(a1))
        return a2

class VAE(Module):

    '''
    the class VAE will initialize an encoder and decoder.
    Encoder: A neural network that takes data x as input, output a vector with dimension 2d, where d is the dimension of latent space
    Decoder: A neural network that samples a latent vector z = \mu(x) + \sigma(x) * epsilon, where epsilon is sampled from a normal distribution N(0, I), output a vector with dimension dim(x)
    '''
    def __init__(self, data_dim, latent_dim):
        encoder = NeuralNetwork(latent_dim * 2, data_dim)
        decoder = NeuralNetwork(data_dim, latent_dim)

    # Randomly sample a z from gaussian distribution with mean \mu(x) and variance \sig(x)
    def sample(self, mean, variance):
        z = np.random.multivariate_normal(mean, variance)
        return z
    
    def forward(self, x):
        dist = self.encoder.forward(x) # Tensor with dimension latent_dim * 2, batch_size
        