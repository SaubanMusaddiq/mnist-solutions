import numpy as np
import matplotlib.pyplot as plt
import mnist
import keras
from keras.layers import Activation, Input, Dense
from keras.utils import to_categorical
from keras.models import Sequential

def ann_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_mnist_data():
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    x_test = mnist.test_images()
    y_test = mnist.test_labels()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = (x_train / 255) - 0.5
    x_test = (x_test / 255) - 0.5

    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist_data()
m,n = x_train.shape

model = ann_model(n)
model.summary()
model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=32)

# evaluate the keras model
_, accuracy = model.evaluate(x_test, to_categorical(y_test))
print('Accuracy: %.2f' % (accuracy*100))

# test accuracy 96.05 with only 10 fast epochs
