import numpy as np
import sys
sys.path.append("./..")
from NN import *

DEBUG = True

layersWeights = [
    [[0.1, 0.2, 0.2], [0.1, 0.1, 0.1]],
    [[.4, .1, .2]]
]
myNN = NN(layersWeights)
print(myNN.predict([0.1, 0.1]))

layersWeights = [
    [[-.3, .1, .0], [0.2, 0.1, 0.4]],
    [[.3, .0, -.1], [0.4, 0, .4], [0.1, 0.3, 0.0]],
    [[0.3, -.1, 0.5, -.1], [0.4, 0.3, -.2, .4]]
]
myNN = NN(layersWeights)
print(myNN.predict([-0.3, 0.5]))