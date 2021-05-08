import numpy as np
from sklearn.datasets import load_boston

np.random.seed(42)


class Activation:

    def __init__(self):
        self.z = np.array([])
        self.a = np.array([])


class Sigmoid(Activation):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, z):
        self.z = z
        self.a = 1.0 / (1 + np.exp(-self.alpha * self.z))
        return self.a

    def backward(self):
        local_derivative = self.alpha * (self.a * (1 - self.a))
        return local_derivative


class Identity(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        self.z = z
        self.a = z
        return self.a

    def backward(self):
        local_derivative = np.ones(self.z.shape)
        return local_derivative


class Softmax(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.a = probabilities

        return self.a

    def backward(self):
        pass


class ReLu(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        self.z = z
        self.a = np.maximum(0, z)
        return self.a

    def backward(self):
        local_derivative = np.where(self.z > 0, 1, 0)

        return local_derivative


class Tanh(Activation):

    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, z):
        self.z = z
        self.a = self.alpha * (2 / (1 + np.exp(-2 * (self.beta * self.z))) - 1)
        return self.a

    def backward(self):
        local_derivative = self.beta / self.alpha * (self.alpha ** 2 - self.a ** 2)
        return local_derivative


class LeakyReLu(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        self.z = z
        self.a = np.where(self.z > 0, self.z, 0.01 * self.z)
        return self.a

    def backward(self):
        local_derivative = np.where(self.z > 0, 1, 0.01)

        return local_derivative


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X = X[2:4, :2]
    y = y[:2, np.newaxis]

    print("\nSigmoid")
    activation = Sigmoid()
    print(activation.forward(X[:2]))
    print(activation.backward())

    print("\nReLu")
    activation = ReLu()
    print(activation.forward(X[:2]))
    print(activation.backward())

    print("\nTanh")
    activation = Tanh()
    print(activation.forward(X[:2]))
    print(activation.backward())

    print("\nSoftmax")
    activation = Softmax()
    print(activation.forward(X[:2]))
    print(activation.backward())

    print("\nIdentity")
    activation = Identity()
    print(activation.forward(X[:2]))
    print(activation.backward())

    print("\nLeaky ReLu")
    activation = LeakyReLu()
    print(activation.forward(X[:2]))
    print(activation.backward())
