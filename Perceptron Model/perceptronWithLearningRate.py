import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('Single Layer Perceptron Dataset.csv')

sumOfError = []

x = df.iloc[:8, 1:4].values
y = df.iloc[:8, -1].values

weights = np.zeros(x.shape[1])
bias, learningRate, eposhs = 0, 0.1, len(x)-1

def step(x): 
    return 1 if x >= 0 else 0

def perceptron(x, y, weights, bias): 
    for eposh in range(eposhs):
        print(f'\nEposh {eposh+1}')
        
        # track errors for this epoch
        noOfError = []

        for i in range(len(x)):
            netInput = np.dot(x[i], weights) + bias
            output = step(netInput)
             
            error = y[i] - output
            
            # use list append (np.append returns a new array and was not assigned)
            noOfError.append(error)

            weights = weights + learningRate * error * x[i]
            bias = bias + learningRate * error
             
            print(f'Input: {x[i]}, Target: {y[i]}, Output: {output}, Error: {error}')
            print(f'Updated Weights: {weights}, Bias: {bias}')
        # record total absolute errors for this epoch
        sumOfError.append(int(np.sum(np.abs(noOfError))))
    return weights, bias

weights, bias = perceptron(x, y, weights, bias)

print("Training Completed!")
print("Final Weights:", weights)
print("Final Bias:", bias)

print("\nTesting Perceptron model after learning")
print(sumOfError)

z = df.iloc[8:13, 1:4].values
for i in range(len(z)):
    netInput = np.dot(z[i], weights) + bias
    output = step(netInput)
    print(f'Input: {z[i]} -> Predicted Output: {output}')

plt.plot(range(len(sumOfError)), sumOfError)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.show()