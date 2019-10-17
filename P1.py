import numpy as np

# Leser datasett, legger data i trening- og testsett
dataSet = np.loadtxt("ds-1.txt", unpack=False)

trainingSet = np.zeros(dtype=float, shape=(int(dataSet.shape[0]/2), 5))
j = 0
for i in np.arange(0, 300, 2):
    trainingSet[j] = dataSet[i]
    j += 1

testSet = np.zeros(dtype=float, shape=(int(dataSet.shape[0] / 2), 5))
j = 0
for i in np.arange(1, 300, 2):
    testSet[j] = dataSet[i]
    j += 1

print(trainingSet[0:10])
print(testSet[0:10])
print(dataSet[0:10])
