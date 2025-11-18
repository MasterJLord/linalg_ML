from fileManipulations import *


def forwardPropagation(inputs : np.array, activationFunctions: np.array, weights : np.array, biases : np.array):
    lastStepOutputs = inputs
    for layerNum in range(len(weights)):
        nextStepOutputs = weights[layerNum] * lastStepOutputs
        nextStepOutputs += biases[layerNum]
        for i in range(len(activationFunctions[layerNum])):
            nextStepOutputs[i] = activationFunctions[layerNum](nextStepOutputs[i])
        lastStepOutputs = nextStepOutputs
    return lastStepOutputs
