from batchtrain import *

A = BatchTrainer(sample_size=300, max=100000, closest_atoms=10, iter=100)
# A.train('H')
A.train_on_data('accuracies/perceptron_300s_100000c_100i_10atoms')

B = BatchTrainer(sample_size=300, max=100000, closest_atoms=30, iter=1000)
B.train_on_data('accuracies/perceptron_300s_100000c_1000i_30atoms')