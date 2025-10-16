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

tensor = tf.constant([0.0, 1, 1])
print(tf.nn.softmax(tensor))
print(tensor)
print(tf.size(tensor))
constant = tf.constant([2.0])
print(tensor * constant)
print(constant.shape)
print(tf.size(constant))