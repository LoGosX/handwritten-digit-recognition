import numpy as np
from neural_network import OneLayerNeuralNetwork
from math import sqrt
from load_image_data import load_data, DATASETS_PATH
import visualize
import time

def main():
    

    #load train and test data
    print('Loading test and train data from "%s".' % DATASETS_PATH)
    x_train, y_train, x_test, y_test = load_data()
    
    #reduce number of training examples to speed up the calculations
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    #display it to check if everything is ok
    print('Visualizing first 25 examples')
    visualize.display_data(x_train, 5, 5)
    

    #1 pick a network architecture
    number_of_inputs = 28 * 28 #dimensions of the images
    number_of_outputs = 10 #10 classes - digits from 0 to 9
    number_of_hidden_units = 10 * 10 #number of hidden units in 1'st (and, for now, only) layer
    epsilon = sqrt(6)/sqrt(number_of_inputs + number_of_outputs) #epsilon used to initialize weights in NN. Every connection will be randomly initialized by a number from range [-epsilon, epsilon]
    
    print("Creating a NN with:\n{} input units\n{} output units\n{} hidden layers with {} hidden units each (not including a bias unit)\nepsilon used to initialize weights: {}".format(number_of_inputs, number_of_outputs, 1, number_of_hidden_units, epsilon))
    #Neural Network used to train data
    nn = OneLayerNeuralNetwork(num_inputs = number_of_inputs, num_outputs = number_of_outputs, num_hidden = number_of_hidden_units, epsilon = epsilon)

    #transforming each training and test example to 1D
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    
    #transforming each label to a vector of 0's, where vector[label] = 1
    #TODO: vectorize the loop
    tmp = np.zeros((y_train.size, number_of_outputs))
    for i in range(tmp.shape[0]):
        tmp[i, int(y_train[i])] = 1
    y_train = tmp
    
    print("Shape of training set: {}x{}\nShape of training labels: {}x{}".format(*x_train.shape, *y_train.shape))
    #training neural network
    start = time.perf_counter()
    nn.train(x_train, y_train, 1000, 0.1, 1)
    print("Trained NN in %fs." % (time.perf_counter() - start))

    #make predictions
    predictions = nn.predict(x_test)
    predictions = np.argmax(predictions, 1)
    m = predictions.size
    hits = 0
    misses = 0
    print(predictions.shape, y_test.shape)
    for t in range(m):
        if predictions[t] == y_test[t]:
            hits += 1
        else:
            misses += 1
    print("{} hits, {} misses. {}% accuracy.".format(hits, misses, hits / m * 100))

    #plot cost for each iteration
    visualize.plot_cost(nn.errors)

if __name__ == "__main__":
    main()