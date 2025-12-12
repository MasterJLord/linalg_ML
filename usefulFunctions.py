import math
import numpy as np
import pandas as pd
import random

def L1LossFunction(target : float, output : float):
    return math.abs(target-output)

def L2LossFunction(target : float, output : float):
    return (target - output)**2

def LinearActivation(input : float):
    return input

def LinearPrime(input : float):
    return 1

def ReLUActivation(input : float):
    return max(input, 0)
    
def ReLUPrime(input : float):
    if (input >= 0):
        return 1
    else:
        return 0
    
def LeakyReLUActivation(input : float):
    if (input >= 0):
        return input
    else:
        return input/10
    
def LeakyReLUPrime(input : float):
    if (input >= 0):
        return 1
    else:
        return 0.1


LOSS_FUNCTION_CODES = \
{
    '1': L1LossFunction,
    '2': L2LossFunction
}

ACTIVATION_FUNCTION_CODES = \
{
    '1': LinearActivation,
    '2': ReLUActivation,
    '3': LeakyReLUActivation
}

ACTIVATION_FUNCTION_DERIVATIVES = \
{
    LinearActivation: LinearPrime,
    ReLUActivation: ReLUPrime,
    LeakyReLUActivation: LeakyReLUPrime
}

def ModifyAll(array : np.array, func):
    lowestLevel = array.ndim == 1
    if (lowestLevel):
        for i in range(len(array)):
            array[i] = func(array[i])
    else:
        for nextLevel in array:
            ModifyAll(nextLevel, func)

"""
@return trainingSet, validationSet, testSetIndices|none
"""
def SplitUpDataset(numberOfOptions : int, trainingSetPortion : float, testSetPortion : float = 0):
    trainingSet = []
    validationSet = []
    testSet = []
    trainingPicksRemaining = numberOfOptions * trainingSetPortion
    testPicksRemaining = numberOfOptions * testSetPortion
    totalPicksRemaining = numberOfOptions - 1
    while totalPicksRemaining >= 0:
        if (random.random() <= trainingPicksRemaining / totalPicksRemaining):
            trainingSet.append(totalPicksRemaining)
            trainingPicksRemaining -= 1
        elif (random.random() < testPicksRemaining / totalPicksRemaining):
            testSet.append(totalPicksRemaining)
            testPicksRemaining -= 1
        else:
            validationSet.append(totalPicksRemaining)

        totalPicksRemaining -= 1
    if (testSetPortion > 0):
        return (trainingSet, validationSet, testSet)
    else:
        return (trainingSet, validationSet)


def createTestData(rows : int, eMax : float = 3, varianceMult : float = 0.1, varianceAdd : float = 1) -> np.array:
    xyzLabel = np.random.random((rows, 4))
    for i in xyzLabel:
        i *= eMax
        i[0] = math.exp(i[0])
        i[1] = math.exp(i[1])
        i[2] = math.exp(i[2])
        # Calculates f(x,y,z)
        i[3] = max(0, i[0]-1) + \
            max(0, i[1]-2)/2 + \
            max(0, i[2]-3)/3
        
    if (varianceAdd > 0):
        addModifier = np.random.random((rows, 4))
        addModifier -= 0.5
        addModifier *= (varianceAdd * 2)
    else:
        addModifier = 0
    
    if (varianceMult > 0):
        multModifier = np.random.random((rows, 4))
        multModifier *= varianceMult
        multModifier += 1
    else:
        multModifier = np.ones((rows, 4))

    return addModifier + (xyzLabel * multModifier)
    