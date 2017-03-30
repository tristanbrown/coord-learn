from batchtrain import *

A = BatchTrainer(sample_size=300, max=100000, iter=100)
# A.train('H')
A.train_on_data('accuracies/perceptron_300s_100000c_100i')