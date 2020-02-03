#!/usr/bin/env python3

## First trial for Mnist



from __future__ import print_function
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


from train_model import train_model
from dataSelector import dataSelector

class convnet(train_model):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_test, y_test)
        batch_size = 32
        num_classes = 10
        epochs = 12

        # input image dimensions
        img_rows, img_cols = 28, 28



        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)


        x_train = x_train.astype('float32')
        x_train /= 255

        dtrain_subset= x_train[:10,:,:,:]
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        self.hist = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(self.x_test, self.y_test))
        
        self.model = model

        self.plot("simple_convnet")


if __name__ == '__main__':
        
    # the data, split between train and test sets
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    x_train, y_train = dataSelector(train_data, train_label).findSubset(42)
    x_test, y_test = dataSelector(test_data, test_label).findSubset(8)


    model1 = model1(x_train, y_train)