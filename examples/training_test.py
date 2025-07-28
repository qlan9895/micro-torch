import numpy as np
from tensorflow.keras.datasets import mnist
from core.base import Tensor
from core.functional import F
from core.linear import Linear

# --- Hyperparameters ---
input_dim = 784
hidden_dim = 10
output_dim = 10
lr = 0.01
epochs = 5
batch_size = 64

# --- Load MNIST dataset ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0  # shape: (60000, 784)
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_train_oh = np.eye(10)[y_train]  # one-hot encode labels
y_test_oh = np.eye(10)[y_test]

# --- Model Definition ---
fc1 = Linear(hidden_dim, input_dim)
fc2 = Linear(output_dim, hidden_dim)

def forward(x_batch):
    x = Tensor(x_batch.T, require_grad=False)       # shape: (784, batch)
    z1 = fc1(x)                                      # shape: (128, batch)
    a1 = F.relu(z1)
    z2 = fc2(a1)                                     # shape: (10, batch)
    probs = F.softmax(z2)
    return probs

# --- Training ---
for epoch in range(epochs):
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        x_batch = x_train[batch_idx]
        y_batch = y_train_oh[batch_idx]

        y_true = Tensor(y_batch.T, require_grad=False)  # shape: (10, batch)
        probs = forward(x_batch)
        loss = F.cross_entropy(y_true, probs)

        # Backward and update
        loss.backward()
        for param in {fc1.W, fc1.b, fc2.W, fc2.b}:
            param.data -= lr * param.grad
            param.grad = None
 
        if i % 1024 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.data:.4f}")

# --- Evaluation ---
def evaluate(x_data, y_data):
    correct = 0
    total = x_data.shape[0]

    for i in range(0, total, batch_size):
        x_batch = x_data[i:i + batch_size]
        y_true = y_data[i:i + batch_size]

        probs = forward(x_batch)
        preds = np.argmax(probs.data, axis=0)
        correct += np.sum(preds == y_true)

    acc = correct / total
    print(f"Test Accuracy: {acc:.2%}")

evaluate(x_test, y_test)
