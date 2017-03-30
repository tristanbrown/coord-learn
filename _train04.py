from batchtrain import *

A = BatchTrainer(sample_size=300, max=100000, iter=1000)
# A.train('H')
A.train_on_data('accuracies/perceptron_300s_100000c_1000i')