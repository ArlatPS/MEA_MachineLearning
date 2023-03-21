import numpy as np
my_data = np.genfromtxt('./NN.csv', delimiter=';')

print(my_data.shape)
zeros = np.array([])
ones = np.array([])
for i in range(my_data.shape[0]):
    if my_data[i][0] == 0:
        new_zero = np.array([])
        for j in range(1,my_data.shape[1]):
            new_zero = np.append(new_zero,my_data[i][j])
        print(new_zero.shape)
        zeros = np.append(zeros, new_zero)
print(zeros.shape)