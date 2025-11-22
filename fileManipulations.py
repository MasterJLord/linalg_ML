import csv
from usefulFunctions import *

NEW_LAYER_CODE = '-1'
END_NET_CODE = '-2'


"""
@return (features, labels)
"""
def loadData(fileName : str, labelHeader : str = None):
    reader = csv.reader(open(fileName, 'r', newline=''))
    header = next(reader)
    try:
        labelIndex = header.index(labelHeader)
        header.pop(labelIndex)
    except:
        labelIndex = None
    loadedData = []
    loadedLabels = []
    for row in reader:
        if (labelIndex != None):
            loadedLabels.append(row.pop(labelIndex))
        loadedData.append(row)
    labelsArray = np.array(loadedLabels, dtype=float)
    dataDict = pd.DataFrame(
        columns=header,
        data=np.array(loadedData, dtype=float))
    return (dataDict, labelsArray)
    
    

def storeNeuralNetwork(fileName : str, neuronWeights : np.array, activationFunctions : np.array):
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
            return (listOfListsToListOfArrays(activationFunctions), listOfListsToListOfArrays(weights))
        else:
            activationFunctions[len(activationFunctions)-1].append(ACTIVATION_FUNCTION_CODES[key])
            weights[len(weights)-1].append(next(reader))


def listOfListsToListOfArrays(inputList : list, dtype=None):
    outputList = []
    for oneList in inputList:
        outputList.append(np.array(oneList, dtype=dtype))
    return outputList

