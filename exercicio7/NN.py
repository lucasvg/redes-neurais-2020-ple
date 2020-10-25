class NN:
    """
    weightsMatrix input example: [[[0.1, 0.2, 0.2], [0.1, 0.1, 0.1]]]
    This represents network with 3 inputs, and one layer with 2 neurons and two outputs

    Default activation functions is ReLU. Any neuron's activaction function can be overriden using the method setActivationFunction
    """

    customActivationFunctions = [[]]

    def __init__(self, weightsMatrix):
        self.weightsMatrix = weightsMatrix

    def setActivationFunction(self):
        print("setActivationFunction")

    def predict(self, input):
        for layer in self.weightsMatrix:
            nextInput = []
            for idxNeuron in range(len(layer)):
                integration = 0
                neuronEdges = layer[idxNeuron]
                if(len(input)!=len(neuronEdges)):
                    raise ArithmeticError("Something went wront! (Input amount "+str(input)+" doesnt match neuron's connections amount "+str(neuronEdges))
                for idxEdge in range(len(neuronEdges)):
                    integration += neuronEdges[idxEdge] * input[idxEdge]
                activationFunction = self.ReLU
                nextInput.append(activationFunction(integration))
            input = nextInput
        return input
            
    def ReLU(self, value):
        return value if value > 0 else 0