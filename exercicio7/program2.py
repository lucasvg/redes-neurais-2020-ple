import numpy as np
import sys
sys.path.append("./..")
from NN import *

DEBUG = True

layersWeights = [
    [[0.1, 0.2, 0.2], [0.1, 0.1, 0.1]],
    [[.4, .1]]
]
myNN = NN(layersWeights)
print(myNN.predict([0.1, 0.1, 0.1]))