import numpy as np
my_data = np.genfromtxt('./NN.csv', delimiter=';')

print(my_data.shape)
zeros = np.zeros((190,120))
zeros_count = -1;

ones = np.zeros((31,120))
ones_count = -1
for i in range(my_data.shape[1]):
    if my_data[0][i] == 0:
        zeros_count += 1
        for j in range(1,my_data.shape[0]):
           zeros[zeros_count,j-1] = my_data[j][i]
    if my_data[0][i] == 1:
        ones_count += 1
        for j in range(1,my_data.shape[0]):
           ones[ones_count,j-1] = my_data[j][i]


print(zeros)
print(ones)

# take 10% + 1 of data for testing
zeros_tst = zeros[0:20]
ones_tst = ones[12:16]

# data for learning shuffled 
data = np.concatenate((zeros[20:], ones[0:12], ones[16:]))
labels = np.concatenate((np.zeros(170), np.ones(27)))