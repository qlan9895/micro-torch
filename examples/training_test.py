import numpy as np
from tensorflow.keras.datasets import mnist
from core.base import Tensor
from core.functional import F
from core.linear import Linear
from neural_network import NeuralNetwork

input_dim = 784
output_dim = 10
lr = 0.01
epochs = 5
batch_size = 64

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0  # shape: (60000, 784)
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

y_train_oh = np.eye(10)[y_train]  # one-hot encode labels
y_test_oh = np.eye(10)[y_test]

def train():
    nn = NeuralNetwork(output_dim, input_dim)

    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        total_loss = 0
        steps = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train_oh[batch_idx]

            x_true = Tensor(x_batch.T, require_grad=True)
            y_true = Tensor(y_batch.T, require_grad=False)  # shape: (10, batch)
            probs = nn.forward(x_true)
            loss = F.cross_entropy(y_true, probs)
            total_loss += loss.data
            steps += 1

            loss.backward()
            for param in {nn.l1.W, nn.l1.b, nn.l2.W, nn.l2.b, nn.l3.W, nn.l3.b}:
                param.data -= lr * param.grad
                param.grad = None
        
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / steps}")

    return nn

nn = train()

def evaluate(x_data, y_data):
    correct = 0
    total = x_data.shape[0]

    for i in range(0, total, batch_size):
        x_batch = x_data[i:i + batch_size]
        y_true = y_data[i:i + batch_size]

        x_true = Tensor(x_batch.T, require_grad=True)
        probs = nn.forward(x_true)
        preds = np.argmax(probs.data, axis=0)
        correct += np.sum(preds == y_true)

    acc = correct / total
    print(f"Test Accuracy: {acc:.2%}")

evaluate(x_test, y_test)

