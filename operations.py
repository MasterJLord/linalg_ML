from fileManipulations import *

def CreateModel(numInputs : int, neuronsPerLayer : list, activationFunctions : list):
    pass # might not get around to making this and just stick to the testNet for now
    # Probably will if I want to make a larger neural network than the one I'm currently working with, which I almost certainly will

def ForwardPropagation(inputs : np.array, activationFunctions: np.array, weights : np.array, verbose : bool = False):
    if (verbose):
        intermediateSteps = []
    lastStepOutputs = np.append(1, inputs)
    for layerNum in range(len(weights)):
        nextStepOutputs = np.matmul(weights[layerNum], lastStepOutputs)
        for i in range(len(activationFunctions[layerNum])):
            nextStepOutputs[i] = activationFunctions[layerNum][i](nextStepOutputs[i])
        if (verbose):
            intermediateSteps.append(nextStepOutputs)
        lastStepOutputs = np.append(1, nextStepOutputs)
    if (verbose):
        return intermediateSteps
    else:
        return lastStepOutputs[1]


def BackPropagation(inputs : np.array, label : float, activationFunctions : np.array, weights : np.array, learningRate : float, lossFunction : str = "L2"):
    allOutputs = ForwardPropagation(inputs, activationFunctions, weights, True)
    outputLosses = []
    # Find derivative of loss with respect to output
    if (lossFunction == "L2"):
        activationFunc = activationFunctions[len(activationFunctions) - 1][0]
        activationFuncDerivative = ACTIVATION_FUNCTION_DERIVATIVES[activationFunc]
        outputMultiple = (activationFunc(allOutputs[len(allOutputs)-1][0])-label) * activationFuncDerivative(allOutputs[len(allOutputs)-1][0])
        outputLosses.append(outputMultiple * weights[len(weights)-1][0])
    else:
        raise("Invalid loss function")
    # Find derivative of loss with respect to output
    for layerIndex in range(len(weights)-1, 0, -1): 
        # iterates over every layer except for the first one, and finds the losses for the layer that precedes each of them
        layerOutputLosses = np.zeros(len(weights[layerIndex-1])+1)
        for neuronNum in range(len(weights[layerIndex])):
            activationFuncDerivative = ACTIVATION_FUNCTION_DERIVATIVES[activationFunctions[layerIndex][neuronNum]]
            lossMultiplier = outputLosses[0][neuronNum] * activationFuncDerivative(allOutputs[layerIndex][neuronNum])
            layerOutputLosses += lossMultiplier * weights[layerIndex][neuronNum]
        outputLosses.insert(0, layerOutputLosses[1:]) # [1:] is to exclude biases
    # Find derivative of loss with respect to weights and update weights accordingly
    for layerIndex in range(len(weights)):
        for neuronNum in range(len(weights[layerIndex])):
            activationFunc = activationFunctions[layerIndex][neuronNum]
            activationFuncDerivative = ACTIVATION_FUNCTION_DERIVATIVES[activationFunc]
            lossMultiplier = outputLosses[layerIndex][neuronNum] * activationFuncDerivative(allOutputs[layerIndex][neuronNum]) * learningRate
            for weight in range(len(weights[layerIndex][neuronNum])):
                relevantOutput = 1 if weight==0 else (inputs if layerIndex == 0 else allOutputs[layerIndex-1])[weight-1]
                weights[layerIndex][neuronNum][weight] -= activationFunc(relevantOutput) * lossMultiplier