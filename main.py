from operations import *
import time

(features, labels) = loadData("testData.csv", "label")
(activationFunctions, weights) = loadNeuralNet("testNet.csv")
(trainingSet, validationSet, testSet) = SplitUpDataset(len(labels), 0.8, 0.1)

loss = math.inf
LOSS_FUNCTION = L2LossFunction
TRAINING_COMPLETE_THRESHOLD = 1
MAX_EPOCHS = 1000

epochsElapsed = 0
startTime = time.perf_counter()

while loss > TRAINING_COMPLETE_THRESHOLD and epochsElapsed < MAX_EPOCHS:
    epochsElapsed += 1
    random.shuffle(trainingSet)
    for i in trainingSet:
        BackPropagation(features[i], labels[i], activationFunctions, weights, 0.000001, LOSS_FUNCTION)
    
    loss = 0
    for i in validationSet:
        predictedValue = ForwardPropagation(features[i], activationFunctions, weights)
        loss += LOSS_FUNCTION(labels[i], predictedValue)
    loss /= len(validationSet)
    print(loss)


endTime = time.perf_counter()
for i in testSet:
    predictedValue = ForwardPropagation(features[i], activationFunctions, weights)
    loss += LOSS_FUNCTION(labels[i], predictedValue)
loss /= len(testSet)

print("Test Set Loss: " + str(loss))
print("Epochs elapsed: " + str(epochsElapsed))
print("Time taken: " + str(endTime - startTime) + " seconds")