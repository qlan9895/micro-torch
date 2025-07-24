import numpy as np
from tensorflow.keras.datasets import mnist
from core.Layer import Layer, F
from core.Loss import LossF
from simple_neural_network import NeuralNetwork
import matplotlib.pyplot as plt

# 1) Load & preprocess MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28).T / 255.0   # shape (784, 60000)
X_test  = X_test.reshape(-1, 28*28).T / 255.0
# one‐hot encode labels
def one_hot(labels, C=10):
    Y = np.zeros((C, labels.size))
    Y[labels, np.arange(labels.size)] = 1
    return Y
y_train = one_hot(y_train)
y_test  = one_hot(y_test)

input_dim  = 784
hidden     = 64
output_dim = 10
nn = NeuralNetwork(output_dim, input_dim, hidden)

epochs     = 10
batch_size = 128
lr         = 0.001

X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True) + 1e-8  # add epsilon to avoid division by 0
X_train_norm = (X_train - X_mean) / X_std

def train():
    num_samples = X_train_norm.shape[1]
    for epoch in range(1, epochs+1):
        perm = np.random.permutation(num_samples)
        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            Xb = X_train_norm[:, idx]     # (784, batch)
            yb = y_train[:, idx]     # (10,  batch)

            # forward
            y_pred = nn.forward(Xb)

            # compute loss
            loss = nn.loss(yb, y_pred)
            epoch_loss += loss * Xb.shape[1]

            # backward (fills each layer’s .activation_grad with dW)
            nn.backward(yb, y_pred)

            # gradient descent parameter update
            nn.l1.W -= lr * nn.l1.W_grad
            nn.l1.b -= lr * nn.l1.b_grad
            nn.l2.W -= lr * nn.l2.W_grad
            nn.l2.b -= lr * nn.l2.b_grad
            nn.l3.W -= lr * nn.l3.W_grad
            nn.l3.b -= lr * nn.l3.b_grad
            # (if you computed bias‐gradients in each layer, update them here too)

        epoch_loss /= num_samples
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

    # 5) Evaluate on test set
    y_test_pred = nn.forward(X_test)
    test_loss   = nn.loss(y_test, y_test_pred)
    pred_labels = np.argmax(y_test_pred, axis=0)
    acc = np.mean(pred_labels == np.argmax(y_test, axis=0))
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {acc*100:.2f}%")

def feed_picture():
    # Pick an image from the test set
    img = X_test[:, 10]              # shape (784,)
    label = y_test[:, 10]            # one-hot vector

    # Show image
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {np.argmax(label)}")
    plt.axis('off')
    plt.show()

    img_norm = (img - X_mean[:, 0]) / X_std[:, 0]  # normalize using training mean/std
    output = nn.forward(img_norm.reshape(-1, 1))   # shape (10, 1)
    pred_label = np.argmax(output)
    print(f"Predicted Label: {pred_label}")

train()
feed_picture()