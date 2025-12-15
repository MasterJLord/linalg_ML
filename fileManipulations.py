import csv
from usefulFunctions import *

NEW_LAYER_CODE = '-1'
END_NET_CODE = '-2'


"""
@return (features, labels)
"""
def LoadData(fileName : str, labelHeader : str = None):
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
    # dataDict = pd.DataFrame(
    #     columns=header,
    #     data=np.array(loadedData, dtype=float))
    return (np.array(loadedData, dtype=float), labelsArray)
    
    

def StoreNeuralNetwork(fileName : str, neuronWeights : np.array, activationFunctions : np.array) -> None:
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
@Return: (activation functions, weights)
"""
def LoadNeuralNet(fileName : str, hasWeights : bool = True):
    reader = csv.reader(open(fileName, 'r', newline=''))
    activationFunctionKeys = next(reader)
    if hasWeights:
        return loadNeuralNetWeights(reader, activationFunctionKeys)
    else:
        numFeatures = int(next(reader)[0])
        return createNeuralNetWeights(activationFunctionKeys, numFeatures)


def createNeuralNetWeights(activationFunctionKeys : list, numFeatures : int):
    activationFunctions = []
    weights = []
    for key in activationFunctionKeys:
        if (key == NEW_LAYER_CODE):
            activationFunctions.append([])
        elif (key == END_NET_CODE):
            break
        else:
            activationFunctions[len(activationFunctions)-1].append(ACTIVATION_FUNCTION_CODES[key])
    weights.append(np.random.random((len(activationFunctions[0]), numFeatures + 1))*10)
    for i in range(1, len(activationFunctions)):
        weights.append(np.random.random((len(activationFunctions[i]), len(activationFunctions[i-1]) + 1))*10)
    return (listOfListsToListOfArrays(activationFunctions, None), weights)



def loadNeuralNetWeights(reader : csv.reader, activationFunctionKeys : list):
    activationFunctions = []
    weights = []
    for key in activationFunctionKeys:
        if (key == NEW_LAYER_CODE):
            weights.append([])
            activationFunctions.append([])
        elif (key == END_NET_CODE):
            break
        else:
            activationFunctions[len(activationFunctions)-1].append(ACTIVATION_FUNCTION_CODES[key])
            weights[len(weights)-1].append(next(reader))
    return (listOfListsToListOfArrays(activationFunctions, None), listOfListsToListOfArrays(weights, float))


def listOfListsToListOfArrays(inputList : list, dType=None):
    outputList = []
    for oneList in inputList:
        outputList.append(np.array(oneList, dtype=dType))
    return outputList

