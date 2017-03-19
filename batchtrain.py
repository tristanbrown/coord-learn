## Training using the Perceptron implementation in scikit-learn, as described
## in Python Machine Learning (pg 50), by Sebastian Raschka. 

from moldata import *
import pandas as pd

class BatchTrainer():
    """
    """
    def __init__(self):
        self.Periodic_Table = pd.read_csv('element_data.csv',
                                        delimiter=',', header=0, index_col=0)
        self.NN = Perceptron(n_iter=40, eta0=0.1, random_state=0)
        self.accuracy = [] # Ordered dictionary vs dataframe?
    
    def train(self, element):
        """"""
        

trainset1 = Molset(200, 'C', 20)
print(trainset1.X)
print(trainset1.y)
print([(len(trainset1.X), len(trainset1.X[0])), len(trainset1.y)])

## Training.
# Split the data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        trainset1.X, trainset1.y, test_size=0.3, random_state=0)

# Standardize the data to a normal distribution.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train on the standardized data. 
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# Apply the trained network to the standardized test data and print metrics.
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

## Plotting
import matplotlib.pyplot as plt

