import numpy as np
import matplotlib.pyplot as plt
import mnist
import keras
from keras.layers import Activation, Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.models import Sequential

def cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),  activation="relu", input_shape=input_shape, name='conv0'))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_model_deeper(input_shape):
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=input_shape))
    #model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.4))
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

    x_train = (x_train / 255)
    x_test = (x_test / 255)

    x_train = x_train.reshape((-1,28,28,1))
    x_test = x_test.reshape((-1,28,28,1))
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist_data()

input_shape = x_train.shape

# cnn_model = cnn_model(input_shape[1:])
cnn_model = cnn_model(input_shape[1:])
cnn_model.summary()
cnn_model.fit(x_train, to_categorical(y_train), epochs=3, batch_size=128)

# evaluate the keras model
_, accuracy = cnn_model.evaluate(x_train, to_categorical(y_train))
print('Train Accuracy: %.2f' % (accuracy*100))
_, accuracy = cnn_model.evaluate(x_test, to_categorical(y_test))
print('Test Accuracy: %.2f' % (accuracy*100))

# Without Droupout training accuracy of 99.79, Epochs =7-8, test accuracy = 98.96
from keras.utils import plot_model
plot_model(cnn_model, to_file='cnn_deep_model.png',show_shapes=True, show_layer_names=False)
