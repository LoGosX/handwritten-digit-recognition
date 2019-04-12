import numpy as np
from neural_network import OneLayerNeuralNetwork, MultiLayerNeuralNetwork, gradient_checking
from math import sqrt
from load_image_data import load_data, DATASETS_PATH
import visualize
import time
import matplotlib.pyplot as plt


def main():
    #load train and test data
    print('Loading test and train data from "%s".' % DATASETS_PATH)
    x_train, y_train, x_test, y_test = load_data()
    
    #reduce number of training examples to speed up the calculations
    x_train = x_train[:]
    y_train = y_train[:]


    #display it to check if everything is ok
    print('Visualizing first 25 examples')
    visualize.display_data(x_train, 5, 5)
    plt.show()

    #1 pick a network architecture
    number_of_inputs = 28 * 28 #dimensions of the images
    number_of_outputs = 10 #10 classes - digits from 0 to 9
    number_of_hidden_units = [100, 100] #number of hidden units in 1'st (and, for now, only) layer
    epsilon = sqrt(6)/sqrt(number_of_inputs + number_of_outputs) #epsilon used to initialize weights in NN. Every connection will be randomly initialized by a number from range [-epsilon, epsilon]
    
    print("Creating a NN with:\n{} input units\n{} output units\n{} hidden layers with {} hidden units each (not including a bias unit)\nepsilon used to initialize weights: {}".format(number_of_inputs, number_of_outputs, 1, number_of_hidden_units, epsilon))
    #Neural Network used to train data
    #nn1 = OneLayerNeuralNetwork(num_inputs = number_of_inputs, num_outputs = number_of_outputs, num_hidden = number_of_hidden_units, epsilon = epsilon, random_seed = 0)
    layers = [number_of_inputs, *number_of_hidden_units, number_of_outputs]
    nn1 = MultiLayerNeuralNetwork(layers, epsilon = epsilon, random_seed = 0)
    
    #transforming each training and test example to 1D
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    #x_train = x_train / 255 # "feature scaling"
    #x_test = x_test / 255 # "feature scaling"

    #transforming each label to a vector of 0's, where vector[label] = 1
    #TODO: vectorize the loop
    tmp = np.zeros((y_train.size, number_of_outputs))
    for i in range(tmp.shape[0]):
        tmp[i, int(y_train[i])] = 1
    y_train = tmp
    
    if False:
        print("Gradient checking\nApprox | Calculated by NN")
        grad, approx, mean = gradient_checking(x_train[0:100], y_train[0:100], layers, 0)
        for i in range(approx.size):
            print("[{}]:\t{:5f}\t{:5f}".format(i, approx[i], grad[i]))
        print("Mean difference", mean)
    print("Shape of training set: {}x{}\nShape of training labels: {}x{}".format(*x_train.shape, *y_train.shape))
    #training neural network
    start = time.perf_counter()
    params = (x_train, y_train, 20, 0.3, 0.1)
    nn1.train(*params)
    print("Trained NN in %fs." % (time.perf_counter() - start))

    #make predictions
    hits, misses = nn1.make_predictions(x_test, y_test)
    print("OneLayerNN:\n{} hits, {} misses. {}% accuracy.\n".format(hits, misses, hits / (hits + misses) * 100))    


    #plot cost for each iteration
    visualize.plot_cost(nn1.errors)
    plt.show()
    plt.close()
    
    print("Interactive predictions. Press any key to proceed. Press mouse to stop.")
    m = x_test.shape[0]
    for i in range(m):
        x = x_test[[i]]
        label = y_test[[i]]
        prediction = np.argmax(nn1.predict(x))
        print("Label: {}. Prediction: {}. Click any key to stop.".format(label, prediction))
        visualize.display_digit(x.reshape((28,28)), label, prediction)
        plt.draw()
        if plt.waitforbuttonpress() is False:
            break
            

if __name__ == "__main__":
    main()