import numpy as np
from math import sqrt
from tqdm import trange


class OneLayerNeuralNetwork:
    """
    One layer Neural Network
    """

    def __init__(self, num_inputs, num_hidden, num_outputs, random_seed = None):
        state = np.random.RandomState(random_seed)
        epsilon = 1
        #matrix of weights, used to calculate activation values in layer 2 in respect to values in layer 1 (input)
        self.theta1 = state.rand(num_hidden, num_inputs + 1) * 2 * epsilon - epsilon
        #matrix of weights used to calculate activation values on layer 3 (output layer)
        self.theta2 = state.rand(num_outputs, num_hidden + 1) * 2 * epsilon - epsilon
        self.errors = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, regularization_parameter):
        '''
        Cost function for logistic regression:
            J(theta) = 1/m * [sum(i = 1...m)[sum(k = 1...K)[-y_ik * log(h_x_i)_k - (1-y_ik) * log(1 - h_x_i)_k]]]
        '''
        # make y a m x k matrix (translate each label to vector)
        m = X.shape[0]
        *_, h_x = self.forward_propagation(X)
        return -1/m * np.sum(
                    y * np.log(h_x) + (1-y) * np.log(1 - h_x)
                ) + regularization_parameter / (2 * m) * (np.sum(self.theta1 ** 2) + np.sum(self.theta2 ** 2))

    def theta_grad(self, X, y, regularization_parameter):
        m = y.shape[0] #number of examples
        
        D1 = np.zeros(self.theta1.shape)
        D2 = np.zeros(self.theta2.shape)
        
        #forward propagation for each example
        A1, Z2, A2, Z3, A3 = self.forward_propagation(X)

        #backward propagation for each example
        for t in range(m):
            #column vector by convention
            x = X[[t]].T
            a3 = A3[[t]].T
            a2 = A2[[t]].T
            a1 = A1[[t]].T #just x, but with added bias term
            yt = y[[t]].T
            delta3 = a3 - yt
            delta2 = self.theta2.T @ delta3 * a2 * (1 - a2)
            delta2 = delta2[1:,:] #remove delta2_0
            D1 += delta2 @ a1.T
            D2 += delta3 @ a2.T
        theta1_grad = D1/m
        theta2_grad = D2/m
        #now add regularization part
        theta1_grad[:, 1:] += regularization_parameter / m * self.theta1[:, 1:]
        theta2_grad[:, 1:] += regularization_parameter / m * self.theta2[:, 1:]

        return theta1_grad, theta2_grad

    def forward_propagation(self, X):
        m = X.shape[0]
        
        a1 = np.concatenate([np.ones((m, 1)), X], 1) # m x (n + 1), where n is number of features
        z2 = a1 @ self.theta1.T #theta1 is l_1 x (n + 1), result is m x l_1
        a2 = self.sigmoid(z2)
        a2 = np.concatenate([np.ones((m, 1)), a2], 1) # m x (l_1 + 1)
        z3 = a2 @ self.theta2.T #theta2 is k x (l_1 + 1), result is m x k
        a3 = self.sigmoid(z3)
        return a1, z2, a2, z3, a3

    def train(self, X, y, epochs, learning_rate = 0.1, regularization_parameter = 0.1):
        last_cost = 0
        for i in range(1, epochs + 1):
            theta1_grad, theta2_grad = self.theta_grad(X, y, regularization_parameter)
            self.theta1 -= learning_rate * theta1_grad
            self.theta2 -= learning_rate * theta2_grad
            self.errors.append(self.cost_function(X, y, regularization_parameter))
            print("{}'th iteration. Cost: {:f} ({:f}).".format(i, self.errors[-1], self.errors[-1] - last_cost))
            last_cost = self.errors[-1]
        
    def predict(self, X):
        '''
        X should be a matrix of examples, with each row being one example
        '''
        *_, h_x = self.forward_propagation(X)
        return h_x

    def make_predictions(self, X, y):
        predictions = self.predict(X)
        predictions = np.argmax(predictions, 1)
        m = predictions.size
        hits = 0
        misses = 0
        for t in range(m):
            if predictions[t] == y[t]:
                hits += 1
            else:
                misses += 1
        return hits, misses

    def theta_grad_approx(self, X, y, regularization_parameter):
        grad = [np.zeros(t.shape) for t in [self.theta1, self.theta2]]
        eps = 1e-4
        thetas = [self.theta1, self.theta2]
        for l in trange(thetas):
            for i in range(thetas[l].shape[0]):
                for j in range(thetas[l].shape[1]):
                    thetas[l][i][j] += eps
                    cost_plus = self.cost_function(X, y, regularization_parameter)
                    thetas[l][i][j] -= 2 * eps
                    cost_minus = self.cost_function(X, y, regularization_parameter)
                    thetas[l][i][j] += eps
                    grad[l][i][j] = (cost_plus - cost_minus) / (2 * eps)
        return grad

    def save(self):
        pass