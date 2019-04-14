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
        state = np.random.RandomState(random_seed)
        eps = sqrt(6) / (sqrt(layers[0] + layers[-1]))
        self.thetas = [state.rand(nl, l + 1) * 2 * eps - eps for l, nl in zip(layers, layers[1:])]
        self.layers = layers
        self.errors = []
        self.state = state


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
        for l in trange(len(thetas)):
            for i in trange(thetas[l].shape[0]):
                for j in trange(thetas[l].shape[1]):
                    thetas[l][i][j] += eps
                    cost_plus = self.cost_function(X, y, regularization_parameter, thetas)
                    thetas[l][i][j] -= 2 * eps
                    cost_minus = self.cost_function(X, y, regularization_parameter, thetas)
                    thetas[l][i][j] += eps
                    grad_approx[l][i][j] = (cost_plus - cost_minus) / (2 * eps)
        grad = self.theta_grad(X, y, regularization_parameter)
        for l in range(len(grad)):
            for i in range(grad[l].shape[0]):
                for j in range(grad[l].shape[1]):
                    g = grad[l][i][j]
                    ga = grad_approx[l][i][j]
                    diff = abs(g - ga)
                    diff = 'no <------!!!' if diff >= 0.001 else 'ok'
                    print('({:3d},{:3d},{:3d})\t{:5f}\t{:5f}\t[{}]'.format(l, i, j, g, ga, diff))
        grad = np.hstack([g.flatten() for g in grad])
        grad_approx = np.hstack([g.flatten() for g in grad_approx])
        maks = abs(np.max(grad - grad_approx))
        if maks >= 0.001:
            print("Gradient checking failed")
        else:
            print("Gradient checking succeeded")
        
    def train(self, X, y, epochs, learning_rate, regularization_parameter):
        """
        Perform Stochastic Gradient Descend
        """
        batch = 1000
        m = X.shape[0]
        y = transform_labels(y)
        X = np.hstack((np.ones((m, 1)), X))
        yX = np.hstack((y, X))
        for _ in trange(epochs):
            self.state.shuffle(yX)
            y, X = yX[:, :10], yX[:, 10:]
            x = X[:batch]
            yb = y[:batch]
            theta_grads = self.theta_grad(x, yb, regularization_parameter)
            for theta, grad in zip(self.thetas, theta_grads):
                theta -= learning_rate * grad
            self.errors.append(self.cost_function(x, yb, regularization_parameter))

    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        *_, h_x = self.forward_propagation(X)
        predictions = np.argmax(h_x, 1)
        return predictions
