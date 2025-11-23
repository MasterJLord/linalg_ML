from operations import *

(features, labels) = loadData("testData.csv", "label")
(activationFunctions, weights) = loadNeuralNet("testNet.csv")

for i in range(1000):
    for (feature, label) in zip(features, labels):
        BackPropagation(feature, label, activationFunctions, weights, 0.001)
    if (i%50 == 0):
        print(ForwardPropagation([2, 3, 4], activationFunctions, weights)) # should converge to  1.833
    