from fileManipulations import *

def createModel(numInputs : int, neuronsPerLayer : list, activationFunctions : list):
    pass # might not get around to making this and just stick to the testNet for now
    # Probably will if I want to make a larger neural network than the one I'm currently working with, which I almost certainly will

def forwardPropagation(inputs : np.array, activationFunctions: np.array, weights : np.array, biases : np.array, verbose : bool = False):
    if (verbose):
        intermediateSteps = []
    lastStepOutputs = inputs
    for layerNum in range(len(weights)):
        lastStepOutputs = np.append(1, lastStepOutputs)
        nextStepOutputs = np.matmul(weights[layerNum], lastStepOutputs)
        nextStepOutputs += biases[layerNum]
        for i in range(len(activationFunctions[layerNum])):
            nextStepOutputs[i] = activationFunctions[layerNum](nextStepOutputs[i])
        if (verbose):
            intermediateSteps.append(lastStepOutputs)
        lastStepOutputs = nextStepOutputs
    if (verbose):
        return intermediateSteps
    else:
        return lastStepOutputs
