import numpy as np
from tensorflow.keras.datasets import mnist
from core.NeuralNet import NeuralNetwork

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten 28x28 images to vectors (784,)
X_train = X_train.reshape(-1, 28*28).T / 255.0  # shape: (784, num_samples)
X_test = X_test.reshape(-1, 28*28).T / 255.0

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((num_classes, labels.size))
    one_hot[labels, np.arange(labels.size)] = 1
    return one_hot

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

input_dim = 784
hidden_size = 64   
output_dim = 10

nn = NeuralNetwork(input_dim, output_dim, hidden_size)

def train_minibatch(nn, X_train, y_train, epochs, lr, batch_size):
    num_samples = X_train.shape[1]
    for epoch in range(epochs):
        perm = np.random.permutation(num_samples)  # shuffle data each epoch
        epoch_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            X_batch = X_train[:, batch_idx]
            y_batch = y_train[:, batch_idx]

            y_pred = nn.forward(X_batch)
            loss = nn.MSE_loss(y_batch)
            grads = nn.backward(y_batch, X_batch)

            for key in nn.param:
                nn.param[key] -= lr * grads[key]

            epoch_loss += loss * X_batch.shape[1]  # sum loss weighted by batch size

        epoch_loss /= num_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

train_minibatch(nn, X_train, y_train, epochs=10, lr=0.1, batch_size=128)



