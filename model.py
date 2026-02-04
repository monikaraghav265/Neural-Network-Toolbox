import numpy as np

def train_backprop(X, y, epochs, lr):
    W1 = np.random.randn(X.shape[1], 3)
    W2 = np.random.randn(3, 1)
    loss_list = []

    for i in range(epochs):
        hidden = X @ W1
        output = hidden @ W2

        loss = np.mean((y - output) ** 2)
        loss_list.append(loss)

        W2 += lr * hidden.T @ (y - output)
        W1 += lr * X.T @ ((y - output) @ W2.T)

    return W1, W2, loss_list


def predict(X, W1, W2):
    return (X @ W1) @ W2



