import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('studentDataset.csv')

xTrain = df.iloc[:15, :5].values
yTrain = df.iloc[:15, -1].values

xTrain = (xTrain - xTrain.min()) / (xTrain.max() - xTrain.min())

print(xTrain)

weights = np.zeros(xTrain.shape[1])
bias, learningRate = 0, 0.1

def step(x): 
    return 1 if x >= 0 else 0

noOfPrediction = noOfCorrectPrediction = 0
def perceptron(x, y, weight, bias):
    for eposh in range(len(x)): 
        print(f'Eposh {eposh+1}')
        
        
        for i in range(len(x)):
            netInput = np.dot(x[i], weight) + bias
            error = y[i] - step(netInput)
            
            weight += (learningRate*error*x[i])
            bias += (learningRate*error)
            
            print(f'Input: {np.round(x[i], 2)}, Target: {np.round(y[i])}, Output: {step(netInput)}, Error: {error}')
            print(f'Weight: {np.round(weight, 2)}, Bias: {np.round(bias, 2)}')
            print()
            
            global noOfPrediction, noOfCorrectPrediction
            noOfPrediction += 1
            noOfCorrectPrediction += 1 if error == 0 else 0
    return weight, bias
weights, bias = perceptron(xTrain, yTrain, weights, bias)

# After learning
print('After learning of Model: ')
xTest = df.iloc[-5:, :5].values
xTest = (xTest - xTest.min()) / (xTest.max() - xTest.min())
yTest = df.iloc[-5:, -1].values

for i in range(len(xTest)): 
    netInput = np.dot(xTest[i], weights) + bias
    output = step(netInput)
    print(f'Input: {np.round(xTest[i], 2)}, Output: {np.round(output, 2)}, error: {yTest[i] - output}')
    print(f'Required Output: {yTest[i]}')
    
    
    noOfPrediction += 1
    noOfCorrectPrediction += 1 if step(netInput) == 1 else 0

print(f'Number of prdiction: {noOfPrediction}')
print(f'Number of correct prdiction: {noOfCorrectPrediction}')
print(f'Accuracy of Model: {noOfCorrectPrediction / noOfPrediction}')