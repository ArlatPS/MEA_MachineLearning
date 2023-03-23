import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import numpy as np

# load data
x_trn = np.load('x_trn_aug.npy')
y_trn = np.load('y_trn_aug.npy')
# x_test = np.load('x_tst.npy')
# y_test = np.load('y_tst.npy')
zeros_tst = np.load('zeros_tst.npy')
ones_tst = np.load('ones_tst.npy')


# adding 10% validation data from training (its shuffled)
N = 160

x_val = x_trn[:N]
x_trn = x_trn[N:]
y_val = y_trn[:N]
y_trn = y_trn[N:]

batch_size = 8
epochs = 40

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

score = model.evaluate(zeros_tst ,np.zeros(zeros_tst.shape[0]), verbose=0)
score2 = model.evaluate(ones_tst ,np.ones(ones_tst.shape[0]), verbose=0)


print('Test loss:     ', score[0])
print('Test accuracy: ', score[1])
print('Test loss:     ', score2[0])
print('Test accuracy: ', score2[1])
model.save('mea_NN_model.h5')

# Test loss:      0.03540382906794548
# Test accuracy:  1.0
# Test loss:      1.1150777339935303
# Test accuracy:  0.6666666865348816