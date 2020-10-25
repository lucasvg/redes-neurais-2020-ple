import numpy as np
import sys
sys.path.append("./..")
from NN import *
import math

DEBUG = True

layersWeights = [
    [[-.3, .1, .0], [0.2, 0.1, 0.4]],
    [[.3, .0, -.1], [0.4, 0, .4], [0.1, 0.3, 0.0]],
    [[0.3, -.1, 0.5, -.1], [0.4, 0.3, -.2, .4]]
]
myNN = NN(layersWeights)
print(myNN.predict([-0.3, 0.5]))

myNN.setActivationFunction(0, 0, myNN.sigmoid)
myNN.setActivationFunction(0, 1, lambda x: 1 / (1 + math.exp(-2 * x)))
myNN.setActivationFunction(1, 0, myNN.tanh)
myNN.setActivationFunction(1, 1, myNN.tanh)
myNN.setActivationFunction(1, 2, myNN.tanh)
myNN.setActivationFunction(2, 0, lambda x: x)
myNN.setActivationFunction(2, 1, lambda x: x)

print(myNN.predict([-0.3, 0.5]))