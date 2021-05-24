import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
from MLP.Activations import Sigmoid, ReLu, Tanh, Identity
from MLP.Layers import Output, Dense
from MLP.Loss import MeanSquaredError
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Sequential:

    def __init__(self, epochs=5, seed=42, batch_size=3, alpha=0.1, mu=1):
        self.seed = seed
        self.alpha = alpha
        self.mu = mu
        np.random.seed(self.seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = []
        self.cost = []
        self.averages = []


    def add(self, layer):
        self.layers.append(layer)

    def sample_batch(self, X, y):
        max_index = len(X)
        if self.batch_size <= max_index:
            indices = np.random.choice(max_index, size=self.batch_size, replace=False)
            sample_X = X[indices]
            sample_y = y[indices]
            return sample_X, sample_y
        else:
            return X, y

    def predict(self, X):
        y_pred = []
        for xi in X:
            # xi = xi[:, np.newaxis]
            for layer in self.layers:
                xi = layer.forward(xi)
            xi = xi[0]
            y_pred.append(xi)

        return np.array(y_pred)

    def forward(self, X, y_true):
        for layer in self.layers:
            X = layer.forward(X)

        cost = self.layers[-1].calculate_cost(y_true)
        self.cost.append(np.mean(cost))
        self.averages.append(np.mean(self.cost))

    def backward(self):
        self.layers.reverse()
        upper_error = self.layers[0].backpropagate_error()
        for layer in self.layers[1:]:
            upper_error = layer.backpropagate_error(upper_error)

        self.layers.reverse()

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.alpha, self.mu)

    def fit(self, X_train, y_train, verbose=True):
        for epoch in range(self.epochs):
            X, y = self.sample_batch(X_train, y_train)
            self.forward(X, y)
            self.backward()
            self.update_weights()
            if epoch % 1000 == 0 and verbose:
                print(f"Cost: {self.cost[-1]:.3f}, Average: {self.averages[-1]:.3f}", end="\n", flush=True)
