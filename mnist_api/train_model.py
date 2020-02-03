
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

class train_model():
    def __init__(self, x_test, y_test):
        self.model = None


        if K.image_data_format() == 'channels_first':
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = keras.utils.to_categorical(y_test, num_classes)
        self.x_test = x_test
        self.y_test = y_test

    def test(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score[1]


    def plot(self, title):
        # Plot training & validation accuracy values
        plt.clf()
        plt.plot(self.hist.history['acc'])
        plt.plot(self.hist.history['val_acc'])
        plt.title(title+"_ModelAccuarcy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(title+"_ModelAccuarcy")

        plt.clf()

        # Plot training & validation loss values
        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.title(title+'_ModelLoss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(title+'_ModelLoss')
