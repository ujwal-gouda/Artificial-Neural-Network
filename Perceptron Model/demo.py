import numpy as np

# AND dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0
learning_rate = 1
epochs = 10

# Step activation function
def step_function(z):
    return 1 if z >= 0 else 0

# Training
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        z = np.dot(weights, X[i]) + bias
        y_pred = step_function(z)

        error = y[i] - y_pred
        total_error += abs(error)

        # Update rule
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

    print(f"Epoch {epoch+1}, Error = {total_error}")

print("\nFinal Weights:", weights)
print("Final Bias:", bias)

# Testing
print("\nPredictions:")
for i in range(len(X)):
    z = np.dot(weights, X[i]) + bias
    print(X[i], "->", step_function(z))
