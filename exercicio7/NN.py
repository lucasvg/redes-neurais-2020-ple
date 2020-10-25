class NN:
    """
    weightsMatrix input example: [[[0.1, 0.2, 0.2], [0.1, 0.1, 0.1]]]
    This represents network of:
        2 inputs
        1 layer of 2 neurons
        2 outputs

    Default activation functions is ReLU. Any neuron's activaction function can be overriden using the method setActivationFunction
    """
    DEBUG = False
    customActivationFunctions = [[]],

    def __init__(self, weightsMatrix):
        self.weightsMatrix = weightsMatrix
        self.biases = [1 for i in range(len(weightsMatrix))]

    def setActivationFunction(self):
        print("setActivationFunction")

    def predict(self, input):
        for idxLayer in range(len(self.weightsMatrix)):
            layer = self.weightsMatrix[idxLayer]
            nextInput = []
            for idxNeuron in range(len(layer)):
                neuronEdges = layer[idxNeuron]
                
                if(len(input)+1!=len(neuronEdges)):
                    raise ArithmeticError("Something went wront! (Input amount "+str(input)+" + 1 bias doesnt match neuron's connections amount "+str(neuronEdges))
                
                integration = neuronEdges[0]*self.biases[idxLayer]
                for idxEdge in range(len(input)):
                    if(self.DEBUG):
                        print(neuronEdges[idxEdge+1], input[idxEdge], neuronEdges[idxEdge+1] * input[idxEdge])
                    integration += neuronEdges[idxEdge+1] * input[idxEdge]
                activationFunction = self.ReLU
                nextInput.append(activationFunction(integration))
            input = nextInput
            if(self.DEBUG):
                print(str(idxLayer) + " layer: "+ str(input))
        return input
            
    def ReLU(self, value):
        return value if value > 0 else 0