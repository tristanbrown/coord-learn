## Training using the Perceptron implementation in scikit-learn

import matplotlib.pyplot as plt
from moldata import *
    
# Data prep. 
trainset3 = Molset(1000, 'N', 20)
print(trainset3.X)
print(trainset3.y)
print([(len(trainset3.X), len(trainset3.X[0])), len(trainset3.y)])

## Training Perceptron using scikit-learn, as described in Python Machine
## Learning (pg 50), by Sebastian Raschka. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        trainset3.X, trainset3.y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())