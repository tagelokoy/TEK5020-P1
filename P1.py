import numpy as np
from sklearn import metrics

# Leser datasett, legger data i trening- og testsett
datasetNumber = input("Type number for data set: ")
datasetName = "ds-" + datasetNumber + ".txt"
dataSet = np.loadtxt(datasetName, unpack=False)
length = dataSet.shape[0]
width = dataSet.shape[1]
trainingSet = np.zeros(dtype=float,
                       shape=(int(length/2), width))
testSet = np.zeros_like(trainingSet)

j, k = 0, 0
for i in np.arange(0, length):
    if i % 2 != 0:
        trainingSet[j] = dataSet[i]
        j += 1
    else:
        testSet[k] = dataSet[i]
        k += 1

trueClassLabel = testSet[:,0]

# ------------------------------ Feilrate -------------------------------------
def error_rate(classifications, key_class):
    errors = np.asarray(np.where(classifications != key_class))
    rate = errors.size / classifications.size
    return rate

# ------------------- Minimum feilrate klassifikator --------------------------
# Finner først forventningsvektoren (my) og kovariansmatrisen (zeta)
n1, n2 = 0, 0
pVectorSum1 = np.zeros(shape=(width-1,))
pVectorSum2 = np.zeros_like(pVectorSum1)
for item in trainingSet:
    if item[0] == 1:            # Klasse 1
        n1 += 1
        pVectorSum1 += item[1:width]
    else:                       # Klasse 2
        n2 += 1
        pVectorSum2 += item[1:width]

my1 = 1/n1 * pVectorSum1        # Forventningsvektor for klasse 1
my2 = 1/n2 * pVectorSum2        # Forventningsvektor for klasse 2

sub1 = np.zeros(shape=(width-1, width-1))
sub2 = np.zeros_like(sub1)
for item in trainingSet:
    if item[0] == 1:
        sub1 += np.outer((item[1:width] - my1), (item[1:width] - my1))
    else:
        sub2 += np.outer((item[1:width] - my2), (item[1:width] - my2))
zeta1 = 1/n1 * sub1             # Kovariansmatrise for klasse 1
zeta2 = 1/n1 * sub2             # Kovariansmatrise for klasse 2

# Finner så W, w, og w0 vha. my, zeta
W1 = -0.5*np.linalg.inv(zeta1)
W2 = -0.5*np.linalg.inv(zeta2)

w1 = np.linalg.inv(zeta1).dot(my1)
w2 = np.linalg.inv(zeta2).dot(my2)

w01 = - 0.5*my1.dot(np.linalg.inv(zeta1)).dot(my1)\
      - 0.5*np.log(np.linalg.det(zeta1)) + np.log(n1/dataSet.shape[0])
w02 = - 0.5*my2.dot(np.linalg.inv(zeta2)).dot(my2)\
      - 0.5*np.log(np.linalg.det(zeta2)) + np.log(n2/dataSet.shape[0])

# Tester diskriminantfunksjonen på testsettet, velger klasse 1 dersom g1 > g2
# ellers velges klasse 2 for en gitt egenskapsvektor x.
results = np.zeros(shape=(testSet.shape[0],))
trueClass = np.zeros_like(results)

for i in np.arange(testSet.shape[0]):
    trueClass[i] = testSet[i][0]
    x = testSet[i][1:width]
    g1 = x.dot(W1).dot(x) + w1.dot(x) + w01
    g2 = x.dot(W2).dot(x) + w2.dot(x) + w02
    if g1 > g2:
        results[i] = 1
    else:
        results[i] = 2


print("\n------------------- Minimum feilrate klassifikator --------------------------")
print("\nPredikerte klasser:\n", results)
print("\nFaktiske klasser:\n", trueClass)
print("\nSammenligning: \n",results == trueClass)

compare = results == trueClass
n_feil = 0
for bool in compare:
    if bool == False:
        n_feil+=1
feilrate_minimumFeil = n_feil/compare.size

# ----------------------- Minste kvadraters metode -----------------------------
#Importerer modell for minste kvadaters metode
from sklearn import datasets, linear_model

#Minste kvadraters lineær regresjon
leastSquare = linear_model.LinearRegression()

#Trener modellen med treningssettet
leastSquare.fit(trainingSet[:,1:width], trainingSet[:,0])

#Forutser responsen for testsettet
pred = leastSquare.predict(testSet[:,1:width])
pred_leastSquare = np.round(pred)

print("\n\n----------------------- Minste kvadraters metode -----------------------------")
print("\nPredikerte klasser:\n", pred_leastSquare)
print("\nFaktiske klasser:\n", trueClassLabel)
print("\nSammenligning: \n",pred_leastSquare == trueClassLabel)

compare = pred_leastSquare == trueClassLabel
n_feil = 0;
for bool in compare:
    if bool == False:
        n_feil+=1;
feilrate_leastSquare = n_feil/compare.size

rate1 = error_rate(results, trueClass)
print("Error rate for", datasetName, "using minimum error classification:",
      rate1)

# -------------------- Nærmeste nabo klassifikatoren --------------------------
#Importerer nærmeste nabo klassifikatormodell
from sklearn.neighbors import KNeighborsClassifier

#Lager klassifikator (k = 1)
nearestNeighbour = KNeighborsClassifier(n_neighbors=1, weights='uniform', p=2)

#Trener modellen med treningssettet
nearestNeighbour.fit(trainingSet[:,1:5], trainingSet[:,0])

#Forutser responsen for testsettet
pred_nn = nearestNeighbour.predict(testSet[:,1:5])

print("\n\n-------------------- Nærmeste nabo klassifikatoren --------------------------")
print("\nPredikerte klasser:\n", pred_nn)
print("\nFaktiske klasser:\n", trueClassLabel)
print("\nSammenligning: \n",pred_nn == trueClassLabel)

compare = pred_nn == trueClassLabel
n_feil = 0;
for bool in compare:
    if bool == False:
        n_feil+=1;
feilrate_nn = n_feil/compare.size

#------------------------------ Oppsummering ----------------------------------

print("\n------------------------------ Oppsummering ----------------------------------")

print("\nFeilrateestimat, minimum feilrate:", feilrate_minimumFeil)

print("\nFeilrateestimat, minste kvadraters metode:", feilrate_leastSquare)

print("\nFeilrateestimat, nærmeste nabo:", feilrate_nn)

print("\nNøyaktighet, minimum feilrate, datasett "+datasetNumber+":",metrics.accuracy_score(trueClass, results))

print("\nNøyaktighet, minste kvadraters metode, datasett "+datasetNumber+":",metrics.accuracy_score(trueClassLabel , pred_leastSquare))

print("\nNøyaktighet, nærmeste nabo, datasett "+datasetNumber+":",metrics.accuracy_score(trueClassLabel, pred_nn),"\n")

