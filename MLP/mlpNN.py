import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


df = pd.read_csv('student_pass_fail_large.csv')
x = df.iloc[:, :6].values
x= (x- x.mean()) / x.std()

y = df.iloc[:, -1].values
y = y.reshape(-1, 1)

input_size = 6
hidden_size = input_size*2
output_size = 1

m = x.shape[0]

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 1000


for epoch in range(epochs):

    # Forward Propagation
    z1 = x.dot(W1) + b1
    a1 = relu(z1)

    z2 = a1.dot(W2) + b2
    y_hat = sigmoid(z2)

    # Binary Cross Entropy Loss
    loss = -np.mean(y*np.log(y_hat+1e-8) + (1-y)*np.log(1-y_hat+1e-8))

    # Backpropagation
    dz2 = y_hat - y
    dW2 = a1.T.dot(dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = dz2.dot(W2.T) * relu_derivative(z1)
    dW1 = x.T.dot(dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # Update Weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


predictions = (y_hat > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print("Final Accuracy:", accuracy)