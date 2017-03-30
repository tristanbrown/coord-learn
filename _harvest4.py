from batchtrain import *

A = BatchTrainer(sample_size=300, max=100000)
# A.train('H')
# A.harvest('C')

A.harvest_all(start='Bi', end='Lr')