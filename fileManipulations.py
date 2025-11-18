import csv
from usefulFunctions import *

NEW_LAYER_CODE = '-1'
END_NET_CODE = '-2'

def loadData(fileName : str, featureHeaders : list, labelHeaders : list):
    return 

def storeNeuralNetwork(fileName : str, neuronWeights : np.array, neuronBiases : np.array, activationFunctions : np.array):
    writer = csv.writer(open(fileName, 'w', newline=''))
    activationFunctionKeys = []
    for layer in activationFunctions:
        activationFunctionKeys.append(NEW_LAYER_CODE)
        for func in layer:
            for pair in ACTIVATION_FUNCTION_CODES.items():
                if (pair[1] == func):
                    activationFunctionKeys.append(pair[0])
                    continue
    activationFunctionKeys.append(END_NET_CODE)
    writer.writerow(activationFunctionKeys)
    for layer in neuronWeights:
        writer.writerows(layer)
    writer.writerow(neuronBiases)

"""
@Return: (activation functions, weights, biases)
"""
def loadNeuralNet(fileName : str):
    reader = csv.reader(open(fileName, 'r', newline=''))
    activationFunctionKeys = next(reader)
    activationFunctions = []
    weights = []
    for key in activationFunctionKeys:
        if (key == NEW_LAYER_CODE):
            weights.append([])
            activationFunctions.append([])
        elif (key == END_NET_CODE):
            biases = next(reader)
            biases = splitUpBiases(biases, activationFunctionKeys)
            return (lostOfListsToListOfArrays(activationFunctions), lostOfListsToListOfArrays(weights), lostOfListsToListOfArrays(biases))
        else:
            activationFunctions[len(activationFunctions)-1].append(ACTIVATION_FUNCTION_CODES[key])
            weights[len(weights)-1].append(next(reader))


def splitUpBiases(biases : list, splittingInformation : list):
    outputBiases = []
    bias = iter(biases)
    for instruction in splittingInformation:
        if (instruction == NEW_LAYER_CODE):
            outputBiases.append([])
        elif (instruction == END_NET_CODE):
            return outputBiases
        else:
            outputBiases[len(outputBiases) - 1].append(next(bias))


def lostOfListsToListOfArrays(inputList : list, dtype=None):
    outputList = []
    for oneList in inputList:
        outputList.append(np.array(oneList, dtype=dtype))
    return outputList


print(loadNeuralNet("testNet.csv"))

"""
import csv
from usefulFunctions import *

NEW_LAYER_CODE = '-1'
END_NET_CODE = '-2'

def loadData(fileName : str, featureHeaders : list, labelHeaders : list):
    return 

def storeNeuralNetwork(fileName : str, neuronWeights : np.array, neuronBiases : np.array, activationFunctions : np.array):
    writer = csv.writer(open(fileName, 'w', newline=''))
    activationFunctionKeys = []
    for layer in activationFunctions:
        activationFunctionKeys.append(NEW_LAYER_CODE)
        for func in layer:
            for pair in ACTIVATION_FUNCTION_CODES.items():
                if (pair[1] == func):
                    activationFunctionKeys.append(pair[0])
                    continue
    activationFunctionKeys.append(END_NET_CODE)
    writer.writerow(activationFunctionKeys)
    for layer in neuronWeights:
        writer.writerows(layer)
    writer.writerow(neuronBiases)

def loadNeuralNet(fileName : str):
    reader = csv.reader(open(fileName, 'r', newline=''))
    activationFunctionKeys = next(reader)
    activationFunctions = []
    weights = []
    for key in activationFunctionKeys:
        if (key == NEW_LAYER_CODE):
            weights.append(None)
        elif (key == END_NET_CODE):
            biases = next(reader)
            biases = splitUpBiases(biases, activationFunctionKeys)
            return (np.array(activationFunctions), weights, biases)
        else:
            activationFunctions.append(ACTIVATION_FUNCTION_CODES[key])
            if (weights[len(weights)-1] is None):
                weights[len(weights)-1] = next(reader)
            else:
                weights[len(weights)-1] = np.vstack((weights[len(weights)-1], next(reader)))


def splitUpBiases(biases : list, splittingInformation : list):
    outputBiases = []
    bias = iter(biases)
    for instruction in splittingInformation:
        if (instruction == NEW_LAYER_CODE):
            outputBiases.append([])
        elif (instruction == END_NET_CODE):
            return outputBiases
        else:
            outputBiases[len(outputBiases) - 1].append(next(bias))

(activationFunctions, weights, biases) = loadNeuralNet("testNet.csv")
print(activationFunctions)
print(weights)
print(biases)
"""