import tensorflow as tf
import pandas as pd
import numpy as np

testArray = np.array([["1"], np.array([1])])
print(testArray)
testArray = np.random.randint(1, 4, (2, 3, 4))
print(testArray)
testArray = np.random.random((2, 3,4)) - .5 + testArray
print(testArray)

testFrame = pd.DataFrame(
    data = [[1, 2], [4, 5]], # Can work with non-numpy lists
    columns=("aa", "bb")
)
print(testFrame)
storage = np.array(testFrame.iloc[0:2])
testFrame = pd.DataFrame(
    data = storage
)
print(testFrame[0])

tensor = tf.constant([[0.0, 0.1, 0.2, 0.3], [0, 1, 2, 3]])
print(tensor[0, 1])
print(tensor[0][1]) # same
# testList = [[0.0, 0.1, 0.2, 0.3], [0, 1, 2, 3]]
# print(testList[0][0:3:2])
# print(testList[0, 1]) not allowed with regular lists, only with tensors

print(tf.reshape(tensor, (4, 2)))
print(tf.reshape(tensor, (2, 2, 2)))
print(tensor.numpy())

ragged = tf.ragged.constant([[[1, 1], [1]], [[1, 1, 1, 1, 1]]]) # ragged tensors can be created with incomplete arrays before the final level
print(ragged.numpy())
print(tf.constant([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])) # regular tensors and ragged tensors are stored completely differently, or at least converted into np arrays completely differently