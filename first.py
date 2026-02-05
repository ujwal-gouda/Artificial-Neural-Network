import random
import math

# Activation
def sigmoid(x):
    # numerically stable sigmoid: avoid overflow for large negative x
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1 + ex)

# Training data (AND gate)
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [0, 0, 0, 1]

# Weights
w1 = random.random()
w2 = random.random()
b1 = random.random()

w3 = random.random()
b2 = random.random()

lr = 0.1

# Training
for epoch in range(5000):
    for i in range(4):
        x1, x2 = X[i]
        y = Y[i]

        # ---- Forward ----
        h = sigmoid(w1*x1 + w2*x2 + b1)
        y_hat = sigmoid(w3*h + b2)

        # ---- Backward ----
        error = y_hat - y

        w3 -= lr * error * h
        b2 -= lr * error

        w1 -= lr * error * w3 * x1
        w2 -= lr * error * w3 * x2
        b1 -= lr * error * w3

# Testing
print("Results:")
for x in X:
    h = sigmoid(w1*x[0] + w2*x[1] + b1)
    y_hat = sigmoid(w3*h + b2)
    print(x, "→", round(y_hat, 3))
