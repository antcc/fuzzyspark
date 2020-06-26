import numpy as np
import sys
from sklearn.model_selection import train_test_split

data = np.genfromtxt(sys.argv[1], delimiter = ",")
X = data[:, 0:-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size = 0.7, shuffle = True, stratify = y)

for x, y in zip(X_train, y_train):
    for i in x:
        print(i, end = ",")
    print(y)

for x, y in zip(X_test, y_test):
    for i in x:
        print(i, file = sys.stderr, end = ",")
    print(y, file = sys.stderr)

