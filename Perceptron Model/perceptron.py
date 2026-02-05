import numpy as np

def step(weightedSum, bias): 
    return 0 if weightedSum <= bias else 1

x = np.array([0.1, 0.5, 0.2])
w = np.array([0.4, 0.3, 0.6])
bias = 0.5

weightedSum = np.sum(x * w)
print(weightedSum)

signFun = step(weightedSum, bias)
print(signFun)