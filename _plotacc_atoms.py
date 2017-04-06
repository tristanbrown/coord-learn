import matplotlib.pyplot as plt
import pandas as pd

Data = pd.read_csv('accuracies/perceptron_300s_100000c_100i_10atoms.csv',
                    delimiter=',', header=0, index_col=0)

elements = Data.index
x = range(len(elements))
y = Data['Accuracy']

plt.plot(x, y, color='green', marker='o', markersize=5, label='100 cycles/10 atoms')

Data = pd.read_csv('accuracies/perceptron_300s_100000c_100i.csv',
                    delimiter=',', header=0, index_col=0)

elements = Data.index
x = range(len(elements))
y = Data['Accuracy']

plt.plot(x, y, color='blue', marker='o', markersize=5, label='100 cycles/20 atoms')

Data = pd.read_csv('accuracies/perceptron_300s_100000c_1000i_30atoms.csv',
                    delimiter=',', header=0, index_col=0)

elements = Data.index
x = range(len(elements))
y = Data['Accuracy']

plt.plot(x, y, color='red', marker='o', markersize=5, label='1000 cycles/30 atoms')


plt.xticks(x, elements)
plt.grid()
plt.title('Perceptron Prediction of Coordination Number')
plt.xlabel('Elements')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()