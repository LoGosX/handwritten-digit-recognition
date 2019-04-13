import numpy as np
from one_layer_nn import OneLayerNeuralNetwork
from multilayer_nn import MultiLayerNeuralNetwork
from load_image_data import load_data, DATASETS_PATH
import visualize
import matplotlib.pyplot as plt


GRADIENT_CHECKING = True

def gradient_checking(nn, X, y, regularization_parameter):
    grad = nn.theta_grad(X, y, regularization_parameter)
    grad_approx = nn.theta_grad_approx(X, y, regularization_parameter)
    
    grad = np.concatenate([x.flatten() for x in grad])
    grad_approx = np.concatenate([x.flatten() for x in grad_approx])
    for i in range(grad.shape[0]):
        print("{:5f}\t{:5f}\t{}".format(grad[i], grad_approx[i], '!' if abs(grad[i] - grad_approx[i]) >= 0.001 else "ok" ))
    return np.max(grad - grad_approx)


def main():
    #load train and test data
    print('Loading test and train data from "%s".' % DATASETS_PATH)
    x_train, y_train, x_test, y_test = load_data()
  
    #display it to check if everything is ok
    print('Visualizing first 25 examples')
    visualize.display_data(x_train, 5, 5)
    plt.show()

   
    x_train = x_train / 255
    x_test = x_test / 255

    #transforming each training and test example to 1D
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], 1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], 1)

    #transforming each label to a vector of 0's, where vector[label] = 1
    #TODO: vectorize the loop
    tmp = np.zeros((y_train.shape[0], 10))
    for i in range(tmp.shape[0]):
        tmp[i, int(y_train[i])] = 1
    y_train = tmp

    #1 pick a network architecture
    number_of_inputs = 28 * 28 #dimensions of the images
    number_of_outputs = 10 #10 classes - digits from 0 to 9
    number_of_hidden_units = [16, 16] #number of hidden units in 1'st (and, for now, only) layer
    epochs = 0
    learning_rate = 0.3
    regularization_parameter = 0.1
    layers = [number_of_inputs, *number_of_hidden_units, number_of_outputs]

    print("Creating a NN with:\n{} input units\n{} output units\n{} hidden layers with {} hidden units (not including a bias units)".format(number_of_inputs, number_of_outputs, len(layers) - 2, number_of_hidden_units))
    #Neural Network used to train data
    #nn1 = OneLayerNeuralNetwork(num_inputs = number_of_inputs, num_outputs = number_of_outputs, num_hidden = number_of_hidden_units[0], epsilon = epsilon, random_seed = 0)
    nn1 = MultiLayerNeuralNetwork(layers, random_seed = 0)
    
    if GRADIENT_CHECKING:
        print("Performing gradient checking")
        max_err = gradient_checking(nn1, x_train[0:10], y_train[0:10], regularization_parameter)
        print("Max difference in calculated gradients:", max_err)

    #x_train = x_train / 255 # "feature scaling"
    #x_test = x_test / 255 # "feature scaling"

    if False:
        print("Gradient checking\nApprox | Calculated by NN")
        grad, approx, mean = gradient_checking(x_train[0:100], y_train[0:100], layers, 0)
        for i in range(approx.size):
            print("[{}]:\t{:5f}\t{:5f}".format(i, approx[i], grad[i]))
        print("Mean difference", mean)
    print("Shape of training set: {}x{}\nShape of training labels: {}x{}".format(*x_train.shape, *y_train.shape))


    #training neural network
    params = (x_train, y_train, epochs, learning_rate, regularization_parameter)
    try:
        nn1.train(*params)
    except Exception as e:
        pass

    #plot cost for each iteration
    visualize.plot_cost(nn1.errors)
    plt.show()
    plt.close()
    
    print("Interactive predictions. Press any key to proceed. Press mouse to stop.")
    m = x_test.shape[0]
    for i in range(m):
        x = x_test[[i]]
        label = y_test[i]
        prediction = np.argmax(nn1.predict(x))
        print("Label: {}. Prediction: {}. Click any key to stop.".format(label, prediction))
        visualize.display_digit(x.reshape((28,28)), label, prediction)
        plt.draw()
        if plt.waitforbuttonpress() is False:
            break
            

if __name__ == "__main__":
    main()