import numpy as np
from one_layer_nn import OneLayerNeuralNetwork
from multilayer_nn import MultiLayerNeuralNetwork
from load_image_data import load_data, DATASETS_PATH
import visualize
import matplotlib.pyplot as plt


GRADIENT_CHECKING = False


def main():
    #load train and test data
    print('Loading test and train data from "%s".' % DATASETS_PATH)
    x_train, y_train, x_test, y_test = load_data()
    
    #display it to check if everything is ok
    print('Visualizing first 25 examples')
    visualize.display_data(x_train, 5, 5)
    plt.show()

    x_train = x_train[:60000]
    y_train = y_train[:60000]
    x_train = x_train / 255
    x_test = x_test / 255

    #transforming each training and test example to 1D
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    #1 pick a network architecture
    number_of_inputs = 28 * 28 #dimensions of the images
    number_of_outputs = 10 #10 classes - digits from 0 to 9
    number_of_hidden_units = [100] #number of hidden units in 1'st (and, for now, only) layer
    epochs = 20
    learning_rate = 0.3
    regularization_parameter = 1
    layers = [number_of_inputs, *number_of_hidden_units, number_of_outputs]

    print("Creating a NN with:\n{} input units\n{} output units\n{} hidden layers with {} hidden units (not including a bias units)".format(number_of_inputs, number_of_outputs, len(layers) - 2, number_of_hidden_units))
    nn = MultiLayerNeuralNetwork(layers, random_seed = 0)
    
    if GRADIENT_CHECKING:
        nn.gradient_checking(x_test[:100], y_test[:100], regularization_parameter)

    #x_train = x_train / 255 # "feature scaling"
    #x_test = x_test / 255 # "feature scaling"



    #training neural network
    params = (x_train, y_train, epochs, learning_rate, regularization_parameter)
    try:
        nn.train(*params)
    except Exception as e:
        print(e)
        pass

    #plot cost for each iteration
    visualize.plot_cost(nn.errors)
    plt.show()
    plt.close()
    

    m = x_test.shape[0]
    predictions = nn.predict(x_test)
    hits = 0
    misses = 0
    for i in range(m):
        if predictions[i] == y_test[i]:
            hits += 1
        else:
            misses += 1
    print("Accuracy: {} missed, {} hits. ({}%)".format(misses, hits, hits / (hits + misses) * 100))
    print("Interactive predictions. Press any key to proceed. Press mouse to stop.")
    for i in range(m):
        x = x_test[i]
        label = y_test[i]
        print("Label: {}. Prediction: {}. Click any key to stop.".format(label, predictions[i]))
        visualize.display_digit(x.reshape((28,28)), label, predictions[i])
        plt.draw()
        if plt.waitforbuttonpress() is False:
            break
            

if __name__ == "__main__":
    main()