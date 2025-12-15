import csv
from usefulFunctions import createTestData

def makeDataFile(fileName : str, labels : list, *testDataParameters, **testDataKeywordParameters):
    writer = csv.writer(open(fileName, 'w', newline=''))
    testData = createTestData(*testDataParameters, **testDataKeywordParameters)
    writer.writerow(labels)
    writer.writerows(testData)

makeDataFile("testData.csv", ("x", "y", "z", "label"), 500, varianceMult=0.1, varianceAdd=0.5)
