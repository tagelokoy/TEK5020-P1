import numpy as np

# Leser datasett 1
#ds1 = open("ds-1.txt", "r")

Class, d1, d2, d3, d4 = np.loadtxt("ds-1.txt", unpack=True)

print(d1[1])
