from config import *
import time

(features, labels) = LoadData(TRAINING_DATA_FILE, TRAINING_DATA_LABEL)
(activationFunctions, weights) = LoadNeuralNet(NEURAL_NETWORK_FILE, PRETRAINED_NEURAL_NETWORK)
(trainingSet, validationSet, testSet) = SplitUpDataset(len(labels), 0.8, 0.1)

loss = math.inf
epochsElapsed = 0
startTime = time.perf_counter()

while loss > TRAINING_COMPLETE_THRESHOLD and epochsElapsed < MAX_EPOCHS:
    random.shuffle(trainingSet)
    for i in trainingSet:
        BackPropagation(features[i], labels[i], activationFunctions, weights, LEARNING_RATE, LOSS_FUNCTION)
    
    if (epochsElapsed%10 == 0):
        loss = 0
        for i in validationSet:
            predictedValue = ForwardPropagation(features[i], activationFunctions, weights)
            loss += LOSS_FUNCTION(labels[i], predictedValue)
        loss /= len(validationSet)
        print(loss)

    epochsElapsed += 1


endTime = time.perf_counter()
for i in testSet:
    predictedValue = ForwardPropagation(features[i], activationFunctions, weights)
    loss += LOSS_FUNCTION(labels[i], predictedValue)
loss /= len(testSet)

print("Test Set Loss: " + str(loss))
print("Epochs elapsed: " + str(epochsElapsed))
print("Time taken: " + str(endTime - startTime) + " seconds")

if (not np.isnan(loss)):
    StoreNeuralNetwork(SAVE_TO_FILE, weights, activationFunctions)
