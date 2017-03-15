## Training using the Perceptron implementation in scikit-learn, as described
## in Python Machine Learning (pg 50), by Sebastian Raschka. 

from moldata import *
import matplotlib.pyplot as plt
    
# Data prep. 
trainset1 = Molset(1000, 'N', 20)
print(trainset1.X)
print(trainset1.y)
print([(len(trainset1.X), len(trainset1.X[0])), len(trainset1.y)])

## Training.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        trainset1.X, trainset1.y, test_size=0.3, random_state=0)

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