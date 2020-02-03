from dataSelector import dataSelector
from keras.datasets import mnist
from simple_convnet import convnet
from alexnet import alexnet
from perceptron import mlp
import pickle   
from keras.utils.vis_utils import plot_model



if __name__ == '__main__':
    
    
    # the data, split between train and test sets
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    x_train, y_train = dataSelector(train_data, train_label).findSubset(42)
    x_test, y_test = dataSelector(test_data, test_label).findSubset(8)


    model1 = convnet(x_train, y_train, x_test, y_test)

    plot_model(model1.model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)
    

    pickle.dump(model1, open("models/model1", "wb"))

    model2 = alexnet(x_train, y_train, x_test, y_test)
    plot_model(model2.model, to_file='alexnet_model.png', show_shapes=True, show_layer_names=True)

    pickle.dump(model2, open("models/model2", "wb"))

    model3 = mlp(x_train, y_train, x_test, y_test)
    plot_model(model3.model, to_file='mlp_model.png', show_shapes=True, show_layer_names=True)

    pickle.dump(model3, open("models/model3", "wb"))

