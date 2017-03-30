from batchtrain import *

A = BatchTrainer(sample_size=300, max=25000)
# A.train('H')
# A.harvest('C')

A.harvest_all()