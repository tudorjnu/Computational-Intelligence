import numpy as np
from sklearn.datasets import load_iris, make_classification, make_moons
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# create a function to display the decision boundary
def f(x, b, w1, w2):
    y = (-(b / w2) / (b / w1)) * x + (-b / w2)
    return y


# create colours for each class
def create_colours(y):
    colours = []
    for i in y:
        if i == 1:
            colours.append("red")
        else:
            colours.append("blue")
    return colours


class Perceptron:
    def __init__(self, n_iter=4, seed=42, eta=0.1):
        self.seed = seed
        self.n_iter = n_iter
        self.eta = eta
        self.W = np.array([])

    @staticmethod
    def pad(X):
        X_instances = X.shape[0]
        X = np.insert(X, 0, np.ones(X_instances), axis=1)
        return X

    @staticmethod
    def z(X, W):
        return X.dot(W)

    @staticmethod
    def a(z):
        z = (-1 if z <= 0 else 1)
        return z

    def fit(self, X_train, y_train):
        X_train = Perceptron.pad(X_train)
        X_features = X_train.shape[1]
        self.W = np.zeros((X_features, 1))

        print(f"Weights: ", self.W.flatten())
        for iteration in range(self.n_iter):
            for x, y in zip(X_train, y_train):
                z = Perceptron.z(x, self.W)
                a = Perceptron.a(z)
                self.W += (self.eta * (y - a) * x)[:, np.newaxis]
            print(f"Weights: ", self.W.flatten())

    def predict(self, X):
        X = Perceptron.pad(X)
        z = Perceptron.z(X, self.W)
        a = np.sign(z)

        return a


if __name__ == "__main__":
    # load the data
    X_simple, y_simple = make_classification(n_features=2, n_classes=2, n_redundant=0, n_informative=2, random_state=1)
    # transform the labels into -1 and 1
    y_simple = np.where(y_simple == 0, 1, -1)

    # load the data
    X_difficult, y_difficult = noisy_moons = make_moons(n_samples=100, noise=0.1)
    # transform the labels into -1 and 1
    y_difficult = np.where(y_difficult == 0, 1, -1)

    y_simple_colours = create_colours(y_simple)
    y_difficult_colours = create_colours(y_difficult)

    print("\n--Simple Data Perceptron--\n")
    model_simple = Perceptron()
    model_simple.fit(X_simple, y_simple)
    predicted_simple = model_simple.predict(X_simple)
    accuracy_simple = accuracy_score(y_simple, predicted_simple)

    print("\n--Difficult Data Perceptron--\n")
    model_difficult = Perceptron()
    model_difficult.fit(X_difficult, y_difficult)
    predicted_difficult = model_difficult.predict(X_difficult)
    accuracy_difficult = accuracy_score(y_difficult, predicted_difficult)

    weights_simple = model_simple.W.flatten()[1:]
    bias_simple = model_simple.W.flatten()[0]

    weights_difficult = model_difficult.W.flatten()[1:]
    bias_difficult = model_difficult.W.flatten()[0]

    x_simple = np.linspace(-0.8, 0.5, 50)
    y_simple = f(x_simple, bias_simple, weights_simple[0], weights_simple[1])

    x_difficult = np.linspace(-2.5, 3, 50)
    y_difficult = f(x_difficult, bias_difficult, weights_difficult[0], weights_difficult[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.set_style("whitegrid")
    fig.suptitle('Perceptron Classification', fontsize=16)
    ax1.scatter(X_simple[:, 0], X_simple[:, 1], c=y_simple_colours)
    ax1.plot(x_simple, y_simple, c="black")
    ax1.set_title(f"Accuracy: {accuracy_simple}")
    ax1.axis(True)

    ax2.scatter(X_difficult[:, 0], X_difficult[:, 1], c=y_difficult_colours)
    ax2.plot(x_difficult, y_difficult, c="black")
    ax2.set_title(f"Accuracy: {accuracy_difficult}")

    plt.savefig("Perceptron.png")
    plt.show()
