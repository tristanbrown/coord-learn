## Training using the Perceptron implementation in scikit-learn, as described
## in Python Machine Learning (pg 50), by Sebastian Raschka. 

from moldata import *
import pandas as pd
import numpy as np
import os
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
    def __init__(self, sample_size=100, closest_atoms=20, max=5000,
                    test_split=0.3, iter=100):
        self.Table = pd.read_csv('element_data.csv',
                                        delimiter=',', header=0, index_col=0)
        
        self.samples = sample_size
        self.range = closest_atoms
        self.max = max
        self.split = test_split
        
        self.NN = Perceptron(n_iter=iter, eta0=0.1, random_state=0)
        
    def harvest(self, element):
        """Generates and saves a list of CSD identifiers that will give
        the correct number of samples. 
        """
        folder = 'samples/'
        path = folder + element + '_'
        if not os.path.exists(folder):
            os.makedirs(folder)
        temppath = path + str(self.samples) + 'temp'
        trainset = Molset(self.samples, element, self.max,
                            savename=temppath)
        print(trainset.labels)
        trainset.saveset(path + str(trainset.count))
        os.remove(temppath + '.npy')
    
    def train(self, element):
        """Trains the neural network given by self.NN, attempting to use the 
        given number of samples of the given element from the CSD. If too few
        samples are found, returns False."""
        trainset = Molset(self.samples, element, self.max)
        trainset.prepare_data(element, self.range)
        
        try:
            finalsize = len(trainset.X)
            if finalsize < 10:
                raise ValueError("Error: %d samples is not enough to train." 
                                        % finalsize)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    trainset.X, trainset.y, test_size=self.split, 
                    random_state=0)
                sc = StandardScaler()
                sc.fit(X_train)
                X_train_std = sc.transform(X_train)
                X_test_std = sc.transform(X_test)
                
                self.NN.fit(X_train_std, y_train)
                y_pred = self.NN.predict(X_test_std)
                
                accuracy = accuracy_score(y_test, y_pred)
                print('Misclassified samples: %d' % (y_test != y_pred).sum())
                print('Accuracy: %.3f' % accuracy)
                
                return (accuracy, finalsize)
        except ValueError:
            print('Error: Training on %s failed for some reason '
                    '(e.g. not enough samples, only one class label, etc.).' 
                    % element)
            return (np.nan, finalsize)
        
    def train_all(self, filename):
        outputs =  pd.DataFrame(self.Table.index.map(self.train).tolist(), 
                                    columns=['Accuracy', 'Samples'],
                                    index=self.Table.index)
        file = filename + '.csv'
        outputs.to_csv(path_or_buf=file)
