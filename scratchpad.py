import numpy as np
import pandas as pd

testArray = np.array([["1"], np.array([1])])
print(testArray)
testArray = np.random.randint(1, 4, (2, 3, 4))
print(testArray)
# testArray += np.random.random((2, 3,4)) - .5
# testArray = np.array([1, [2, 2], [3, 3]])
# print(testArray)

testFrame = pd.DataFrame(
    data = [[1, 2], [4, 5]], # Can work with non-numpy lists
    columns=("aa", "bb")
)
print(testFrame)
storage = np.array(testFrame.iloc[0:2])
print(type(storage))
testFrame = pd.DataFrame(
    data = storage
)
print(testFrame[0])