## Training using the Perceptron implementation in perceptron.py.

import perceptron as nn1
import matplotlib.pyplot as plt
from moldata import *
    
# Data prep. 
trainset3 = Molset(100, 'N', 20)
print(trainset3.X)
print(trainset3.y)
print([(len(trainset3.X), len(trainset3.X[0])), len(trainset3.y)])



# Training.
ppn = nn1.Perceptron(eta=.00001, n_iter=1000)
ppn.fit(trainset3.X, trainset3.y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Training Cycles')
plt.ylabel('Number of misclassifications')
plt.show()
