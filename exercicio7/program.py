import numpy as np
import sys
sys.path.append("./../")
from utils import *

DEBUG = True

if(DEBUG):
    layersCount = 1
    layersWeights = [[[0.1, 0.2, 0.2], [0.1, 0.1, 0.1]]]
else:
    print("Neural Network Program")
    print("Type the number of layers:")
    layersCount = int(input())
    layersWeights = []
    for i in range(layersCount):
        print("Please provide the weights matrix of the layer number", i, ":")
        print("Example of input: [[0.1, 0.2]]")

        aux = input().split("],")
        aux = [i.replace("[", "").replace("]", "").replace(" ", "").split(",") for i in aux]
        aux = [ [float(j) for j in i] for i in aux]
        layersWeights.append(aux)

if(not GetInput("The default activation function is ReLU, do you want to use different functions ? Y/N", ParseBoolInput)):
    print(" 1 - Sigmoid")
    print(" 2 - Tanh")
    print(GetInput("", ParseLayerNeuronFunction))