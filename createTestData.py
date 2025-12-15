import csv
from config import *
def makeDataFile(fileName : str, labels : list, *testDataParameters, **testDataKeywordParameters):
    writer = csv.writer(open(fileName, 'w', newline=''))
    testData = createTestData(*testDataParameters, **testDataKeywordParameters)
    writer.writerow(labels)
    writer.writerows(testData)



makeDataFile(TRAINING_DATA_FILE, ("x", "y", "z", "f(x,y,z)"), 500, eMax = TRAINING_DATA_MAX_EXPONENT, varianceMult=TRAINING_DATA_MULTIPLICATIVE_VARIANCE, varianceAdd=TRAINING_DATA_ADDITIVE_VARIANCE)
print("Data generation complete!")
