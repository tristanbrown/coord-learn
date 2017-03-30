## Training using the Perceptron implementation in scikit-learn, as described
## in Python Machine Learning (pg 50), by Sebastian Raschka. 

from moldata import *
import pandas as pd
import numpy as np
import os
import re
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
        self.sample_folder = 'samples'
        
        self.NN = Perceptron(n_iter=iter, eta0=0.1, random_state=0)
        
    def harvest(self, element):
        """Generates and saves a list of CSD identifiers that will give
        the correct number of samples. 
        """
        path = self.sample_folder + '/' + element + '_'
        if not os.path.exists(self.sample_folder):
            os.makedirs(self.sample_folder)
        temppath = path + str(self.samples) + 'temp'
        trainset = Molset(self.samples, element, self.max,
                            savename=temppath)
        savename = path + str(trainset.count)
        trainset.saveset(savename)
        print('Saved the following CSD entries to /' + savename + '.npy')
        print(trainset.labels)
        os.remove(temppath + '.npy')
    
    def recover(self, element, filename):
        ids = np.load(os.path.join(self.sample_folder, filename))
        trainset = Molset(ids, element)
        count = trainset.count_atoms(element)
        if count < self.samples:
            trainset.count = count
        else:
            trainset.count = self.samples
        return trainset
    
    def harvest_all(self, start='H', end='Lr'):
        """Harvests datasets from the CSD for each element."""
        self.Table[start:end].index.map(self.harvest)
    
    def train(self, element, filename=None):
        """Trains the neural network given by self.NN, attempting to use the 
        given number of samples of the given element from the CSD. If too few
        samples are found, returns False."""
        
        # Acquiring the training data.
        if filename is None:
            trainset = Molset(self.samples, element, self.max)
        else:
            trainset = self.recover(element, filename)
        
        trainset.prepare_data(element, self.range)
        
        # Training.
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
        
    def train_all(self, outfile):
        """Generate new data for all elements of the periodic table and train
        the NN, generating accuracy values and the number of samples."""
        outputs =  pd.DataFrame(self.Table.index.map(self.train).tolist(), 
                                columns=['Accuracy', 'Samples'],
                                index=self.Table.index)
        
    
    def train_on_data(self, outfile):
        """Train the NN using existing data, generating accuracy values and 
        number of samples."""
        outputs =  pd.DataFrame(np.nan, 
                                columns=['Accuracy', 'Samples'],
                                index=self.Table.index)
                                
        path = self.sample_folder
        files = (file for file in os.listdir(path)
                    if os.path.isfile(os.path.join(path, file)))
        for file in files:
            print('Loading ' + file)
            parts = re.split('_|\.', file)
            if parts[1].isdigit():
                data = self.train(parts[0], file)
                outputs.set_value(parts[0], 'Accuracy', round(data[0], 3))
                outputs.set_value(parts[0], 'Samples', data[1])
        
        outfile = outfile + '.csv'
        outputs.to_csv(path_or_buf=outfile)