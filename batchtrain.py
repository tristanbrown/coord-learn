## Training using the Perceptron implementation in scikit-learn, as described
## in Python Machine Learning (pg 50), by Sebastian Raschka. 

from moldata import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

class BatchTrainer():
    """
    Takes in a sample size and a type of neural network. For each element in the
    periodic table, it creates a data set, preps that data set, trains the
    network, and then records the trained network's accuracy on a prepped test
    set. 
    
    The accuracy values are stored in a [ordered dict or dataframe?], so they
    can be saved to a .csv file and plotted (accuracy vs element). 
    """
    def __init__(self, sample_size=100, closest_atoms=20, test_split=0.3):
        self.Table = pd.read_csv('element_data.csv',
                                        delimiter=',', header=0, index_col=0)
        
        self.samples = sample_size
        self.range = closest_atoms
        self.split = test_split
        
        self.NN = Perceptron(n_iter=40, eta0=0.1, random_state=0)
        
        
    
    def train(self, element):
        """"""
        trainset = Molset(self.samples, element, self.range)
        X_train, X_test, y_train, y_test = train_test_split(
            trainset.X, trainset.y, test_size=self.split, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        
        self.NN.fit(X_train_std, y_train)
        y_pred = self.NN.predict(X_test_std)
        
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        
    def train_all(self):
        self.Table['Accuracy'] =  Table.index.map(self.train)
        print(self.Table)
        
        
        # Insert accuracy_score(y_test, y_pred) into NN Accuracy column.



# ## Plotting
# import matplotlib.pyplot as plt

