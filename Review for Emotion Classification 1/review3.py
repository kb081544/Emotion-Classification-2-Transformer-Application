from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Dropout, MaxPool1D, Flatten
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.model_selection import train_test_split
import pickle

EPOCH=3

train_data=np.loadtxt("assignment2.txt", delimiter=',')
data_x = train_data[:, 1:]
data_y = train_data[:, 0]
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

input_shape = (x_train.shape[1], 1)
model = Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(Dense(8, activation='relu', kernel_initializer='glorot_uniform'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(4, activation='relu', kernel_initializer='glorot_uniform'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))
model.add(tf.keras.layers.Dropout(0.2))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint_callback = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min',
                                      verbose=1)
history = model.fit(x_train, y_train, batch_size=8, epochs=EPOCH, validation_data=(x_test, y_test))
predictions = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose=1)
pred_np = np.array(predictions)
print("\nloss= ", score[0], "\n정답률: ", score[1])
# print("예측")
# print(np.round(predictions))
# now=datetime.now()
# model_name = f'trained_per_peak_{int(score[1] * 100)}.h5'
# model.save(model_name)


