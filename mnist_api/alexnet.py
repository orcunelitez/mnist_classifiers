import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

from train_model import train_model


class alexnet(train_model):
    def __init__(self, X_train, Y_train, X_test, Y_test):
        super().__init__(X_test, Y_test)

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train/=255
        X_test/=255


        # Three steps to create a CNN
        # 1. Convolution
        # 2. Activation
        # 3. Pooling
        # Repeat Steps 1,2,3 for adding more hidden layers

        # 4. After that make a fully connected network
        # This fully connected network gives ability to the CNN
        # to classify the samples
        number_of_class = 10
        Y_train = np_utils.to_categorical(Y_train, number_of_class)
        Y_test = np_utils.to_categorical(Y_test, number_of_class)
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2D(32, kernel_size=(3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,kernel_size=(3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())

        # Fully connected layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))

        model.add(Activation('softmax'))


        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

        test_gen = ImageDataGenerator()

        train_generator = gen.flow(X_train, Y_train, batch_size=64)
        test_generator = test_gen.flow(X_test, Y_test, batch_size=64)


        self.hist = model.fit_generator(train_generator, steps_per_epoch=len(Y_train)//64, epochs=100, 
                    validation_data=test_generator, validation_steps=len(Y_test)//64)

        self.model = model
        self.plot("alexnet")


        
