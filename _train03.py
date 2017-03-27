from batchtrain import *

A = BatchTrainer(sample_size=100, max=2000, iter=100)
# A.train('H')
A.train_all('element_accuracies_100s_2000c_1000i')