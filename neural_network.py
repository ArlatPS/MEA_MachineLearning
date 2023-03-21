import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import numpy as np

# load data
x_trn = np.load('x_trn.npy')
y_trn = np.load('y_trn.npy')
x_test = np.load('x_tst.npy')
y_test = np.load('y_tst.npy')

# adding 5% validation data
x_val = x_trn[0:10]
x_trn = x_trn[10:]
y_val = y_trn[0:10]
y_trn = y_trn[10:]

batch_size = 8
epochs = 50

model = Sequential()
model.add(Input(shape=(120,)))
model.add(Dense(200, activation="relu", bias_initializer="glorot_uniform"))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu", bias_initializer="glorot_uniform"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="sigmoid"))

model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=0.1),loss="sparse_categorical_crossentropy", metrics=['accuracy'])

print('Model parameters = %d' % model.count_params())
print(model.summary())

history = model.fit(x_trn, y_trn, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val,y_val))

score = model.evaluate(x_test,y_test, verbose=0)

print('Test loss:     ', score[0])
print('Test accuracy: ', score[1])
model.save('mea_NN_model.h5')