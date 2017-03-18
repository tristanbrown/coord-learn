## Training using the Perceptron implementation in perceptron.py.

from moldata import *
import perceptron as nn
import matplotlib.pyplot as plt
    
# Data prep. 
trainset1 = Molset(100, 'Fe', 20)
print(trainset1.X)
print(trainset1.y)
print([(len(trainset1.X), len(trainset1.X[0])), len(trainset1.y)])



# Training.
ppn = nn.Perceptron(eta=.00001, n_iter=1000)
ppn.fit(trainset1.X, trainset1.y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Training Cycles')
plt.ylabel('Number of misclassifications')
plt.show()
