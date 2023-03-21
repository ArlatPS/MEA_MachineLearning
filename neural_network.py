import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import numpy as np

# load data
x_trn = np.load('x_trn.npy')
y_trn = np.load('y_trn.npy')
# x_test = np.load('x_tst.npy')
# y_test = np.load('y_tst.npy')
zeros_tst = np.load('zeros_tst.npy')
ones_tst = np.load('ones_tst.npy')


# adding 5% validation data from test (its shuffled)
ones_test_count = 13
ones_to_validate = 7
zeros_test_count = 35
zeros_to_validate = 10

x_val = np.concatenate((zeros_tst[0:zeros_to_validate],ones_tst[0:ones_to_validate]))
zeros_tst = zeros_tst[zeros_to_validate:]
ones_tst = ones_tst[ones_to_validate:]
y_val = np.concatenate((np.zeros(zeros_to_validate), np.ones(ones_to_validate)))
batch_size = 8
epochs = 150

model = Sequential()
model.add(Input(shape=(120,)))
model.add(Dense(240, activation="relu", bias_initializer="glorot_uniform"))
model.add(Dropout(0.2))
model.add(Dense(120, activation="relu", bias_initializer="glorot_uniform"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="sigmoid"))

model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=0.1),loss="sparse_categorical_crossentropy", metrics=['accuracy'])

print('Model parameters = %d' % model.count_params())
print(model.summary())

history = model.fit(x_trn, y_trn, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val,y_val))

score = model.evaluate(zeros_tst ,np.zeros(zeros_test_count - zeros_to_validate), verbose=0)
score2 = model.evaluate(ones_tst ,np.ones(ones_test_count - ones_to_validate), verbose=0)


print('Test loss:     ', score[0])
print('Test accuracy: ', score[1])
print('Test loss:     ', score2[0])
print('Test accuracy: ', score2[1])
model.save('mea_NN_model.h5')

# Test loss:      0.03540382906794548
# Test accuracy:  1.0
# Test loss:      1.1150777339935303
# Test accuracy:  0.6666666865348816