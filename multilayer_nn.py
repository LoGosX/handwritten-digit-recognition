import numpy as np
from math import sqrt
from tqdm import tqdm, trange

def unroll_thetas(theta_vector, layers):
    thetas = []
    i = 0
    for l, nl in zip(layers, layers[1:]):
        i_e = (nl + 1) * l
        thetas.append(thetas[i:i_e].reshape((nl + 1, l)))
        i = i_e
    return thetas

def roll_thetas(thetas):
    return np.concatenate([x.flatten() for x in thetas], 1)

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def transform_labels(labels):
    '''
    transforms a number to a vector. E.g 5 to [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    '''
    m = labels.shape[0]
    new_labels = np.zeros((m, 10))
    new_labels[np.arange(0, m).astype(int), labels.astype(int)] = 1
    return new_labels

class MultiLayerNeuralNetwork:

    def __init__(self, layers, random_seed = None):
        state = np.random.RandomState(0)
        eps = sqrt(6) / (sqrt(layers[0] + layers[-1]))
        self.thetas = [state.rand(nl, l + 1) * 2 * eps - eps for l, nl in zip(layers, layers[1:])]
        self.layers = layers
        self.errors = []


    def cost_function(self, X, y, regularization_parameter, thetas = None):
        if thetas is None:
            thetas = self.thetas
        m = X.shape[0]
        *_, h_x = self.forward_propagation(X, thetas)
        cost = -1 / m * np.sum( y * np.log(h_x) + (1-y) * np.log(1 - h_x))
        regularization = regularization_parameter / (2 * m) * sum(np.sum(t[:, 1:] ** 2) for t in thetas)
        return cost + regularization

    def forward_propagation(self, X, thetas = None):
        if thetas is None:
            thetas = self.thetas
        m = X.shape[0]
        a = X
        activations = [a]
        for theta in thetas:
            z = a @ theta.T
            a = sigmoid(z)
            a = np.hstack((np.ones((m, 1)), a))
            activations.append(a)
        activations[-1] = activations[-1][:, 1:]
        return activations

    def theta_grad(self, X, y, regularization_parameter, thetas = None):
        if thetas is None:
            thetas = self.thetas
        m = X.shape[0]
        activations = self.forward_propagation(X, thetas)
        L = len(activations)
        D = [np.zeros(t.shape) for t in thetas]    
        for t in range(m):
            yt = y[[t]].T
            A = [a[[t]].T for a in activations]
            deltas = [None] * L
            deltas[-1] = A[-1] - yt
            for l in range(L - 2, 0, -1):
                delta = thetas[l].T @ deltas[l + 1] * A[l] * (1 - A[l])
                delta = delta[1:, :]
                deltas[l] = delta
            for l in range(L - 1):
                D[l] += deltas[l+1] @ A[l].T 
        theta_grads = [d / m for d in D]
        for grad, theta in zip(theta_grads, thetas):
            grad[:, 1:] += regularization_parameter / m * theta[:, 1:]
        return theta_grads

    def gradient_checking(self, X, y, regularization_parameter):
        thetas = [np.copy(t) for t in self.thetas]
        eps = 1e-4
        m = X.shape[0]
        y = transform_labels(y)
        X = np.hstack((np.ones((m, 1)), X))
        grad_approx = [np.empty(t.shape) for t in thetas]
        for l in range(len(thetas)):
            for i in range(thetas[l].shape[0]):
                for j in range(thetas[l].shape[1]):
                    thetas[l][i][j] += eps
                    cost_plus = self.cost_function(X, y, regularization_parameter, thetas)
                    thetas[l][i][j] -= 2 * eps
                    cost_minus = self.cost_function(X, y, regularization_parameter, thetas)
                    thetas[l][i][j] += eps
                    grad_approx[l][i][j] = (cost_plus - cost_minus) / (2 * eps)
        grad = self.theta_grad(X, y, regularization_parameter)
        grad = np.hstack((g.flatten() for g in grad))
        grad_approx = np.hstack((g.flatten() for g in grad_approx))
        maks = abs(np.max(grad - grad_approx))
        for i in range(grad.shape[0]):
            diff = abs(grad[i] - grad_approx[i])
            diff = 'no <------!!!' if diff >= 0.001 else 'ok'
            print('({:3d},{:3d},{:3d})\t{:5f}\t{:5f}\t[{}]'.format(l, i, j, grad[i], grad_approx[i], diff))
        if maks >= 0.001:
            print("Gradient checking failed")
        else:
            print("Gradient checking succeeded")
        
    def train(self, X, y, epochs, learning_rate, regularization_parameter):
        m = X.shape[0]
        y = transform_labels(y)
        X = np.hstack((np.ones((m, 1)), X))
        for i in trange(epochs):
            theta_grads = self.theta_grad(X, y, regularization_parameter)
            for theta, grad in zip(self.thetas, theta_grads):
                theta -= learning_rate * grad
            self.errors.append(self.cost_function(X, y, regularization_parameter))

    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        *_, h_x = self.forward_propagation(X)
        predictions = np.argmax(h_x, 1)
        return predictions


























"""
class MultiLayerNeuralNetwork:

    def __init__(self, layers, epsilon = None, random_seed = None):
        if epsilon is None:
            epsilon = sqrt(6)/sqrt(layers[0] + layers[-1])
    
        state = np.random.RandomState(random_seed)
        self.errors = []
        self.layers = layers
        thetas = []
        for l, nl in zip(layers, layers[1:]):
            theta = (state.rand(nl, l + 1) * 2 - 1) * epsilon
            thetas.append(theta)
        self.thetas = thetas

    @property
    def theta_vector(self):
        return np.concatenate([t.flatten() for t in self.thetas])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, regularization_parameter, theta_vector = None, thetas = None):
        '''
        Cost function for logistic regression:
            J(theta) = 1/m * [sum(i = 1...m)[sum(k = 1...K)[-y_ik * log(h_x_i)_k - (1-y_ik) * log(1 - h_x_i)_k]]]
        '''
        if theta_vector is None:
            theta_vector = self.theta_vector
        if thetas is None:
            thetas = self.thetas
        
        # make y a m x k matrix (translate each label to vector)
        m = X.shape[0]
        h_x = self.predict(X, thetas)
        c1 = -1/m * np.sum(np.sum(
                    y * np.log(h_x) + (1-y) * np.log(1 - h_x)
                ))
        c2 = regularization_parameter * np.sum(np.sum(theta_vector ** 2)) / (2 * m)
        return c1 + c2

    def theta_grad(self, X, y, regularization_parameter, thetas = None):
        if thetas is None:
            thetas = self.thetas
        m = y.shape[0] #number of examples
        D = [np.zeros(t.shape) for t in thetas]
        for t in range(m):
            A = self.forward_propagation(X[[t]])
            A = [a.T for a in A]
            yt = y[[t]].T
            
            L = len(self.layers)
            deltas = [None] * L
            deltas[L - 1] = A[-1] - yt
            for l in range(L-2, 0, -1):
                
                deltas[l] = thetas[l].T @ deltas[l+1] * A[l] * (1 - A[l])
                deltas[l] = deltas[l][1:,:]

            for l in range(L-1):
                D[l] += deltas[l+1] @ A[l].T

        theta_grads = [d / m for d in D]
        for theta_grad, theta in zip(theta_grads, thetas):
            theta_grad[:, 1:] += regularization_parameter / m * theta[:, 1:]
        return theta_grads

    def forward_propagation(self, X, thetas = None):
        m = X.shape[0]
        if thetas is None:
            thetas = self.thetas
        a = np.concatenate([np.ones((m, 1)), X], 1)
        ret = [a]
        for theta in thetas:
            z = a @ theta.T
            a = self.sigmoid(z)
            a = np.concatenate([np.ones((m, 1)), a], 1)
            ret.append(a)

        ret[-1] = ret[-1][:, 1:]
        return ret

    def train(self, X, y, epochs, learning_rate = 0.1, regularization_parameter = 1):
        for i in range(epochs):
            theta_grads = self.theta_grad(X, y, regularization_parameter)
            for theta, theta_grad in zip(self.thetas, theta_grads):
                theta -= learning_rate * theta_grad
            self.errors.append(self.cost_function(X, y, regularization_parameter))
            print("{}'th iteration. Cost: {}".format(i + 1, self.errors[-1]))
        
    def predict(self, X, thetas = None):
        '''
        X should be a matrix of examples, with each row being one example
        '''
        *_, h_x = self.forward_propagation(X, thetas)
        return h_x

    def theta_grad_approx(self, X, y, regularization_parameter):
        epsilon = 1e-3
        m = X.shape[0]
        theta_grads = [np.zeros(t.shape) for t in self.thetas]
        for l in range(len(theta_grads)):
            for i in range(theta_grads[l].shape[0]):
                for j in range(theta_grads[l].shape[1]):
                    self.thetas[l][i][j] += epsilon
                    cost1 = self.cost_function(X, y, regularization_parameter)
                    self.thetas[l][i][j] -= 2 * epsilon
                    cost2 = self.cost_function(X, y, regularization_parameter)
                    self.thetas[l][i][j] += epsilon
                    theta_grads[l][i][j] = (cost1 - cost2) / (2 * epsilon) + regularization_parameter * self.thetas[l][i][j]
        return theta_grads

    def save(self):
        theta_vec = np.concatenate([theta.flatten() for theta in self.thetas])
        np.savetxt("thetas.txt", theta_vec)

    def make_predictions(self, X, y):
        predictions = self.predict(X)
        predictions = np.argmax(predictions, 1)
        m = predictions.shape[0]
        hits = 0
        misses = 0
        for t in range(m):
            if predictions[t] == y[t]:
                hits += 1
            else:
                misses += 1
        return hits, misses


def gradient_checking(X, y, layers, regularization_parameter = 0.1):
    nn = MultiLayerNeuralNetwork(layers)
    grad = nn.theta_grad(X, y, regularization_parameter)
    grad_approx = nn.theta_grad_approx(X, y, regularization_parameter)
    grad = np.concatenate([x.flatten() for x in grad])
    grad_approx = np.concatenate([x.flatten() for x in grad_approx])
    return grad, grad_approx, np.mean((grad - grad_approx))
"""