import numpy as np
import math

from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
from Activations import Sigmoid, ReLu, Tanh, Identity, LeakyReLu
from Layers import Output, Dense
from Loss import MeanSquaredError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from Model import Sequential
import time

import matplotlib.pyplot as plt
import seaborn as sns


def plot_four(filename, n_neurons=4, n_layers=1, lr=0.01, batch_size=30, mu=1, epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 12))
    title = "Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for activation, axis in zip([Sigmoid, Tanh, ReLu, LeakyReLu], [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size, mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(activation, n_neurons))
        model.add(Output(Identity, n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        end_time = time.time()
        time_diff = round(end_time - start_time, 2)

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"{activation.__name__} \nAverage Cost: {cost} - Time: {time_diff}s"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def compare_lr(filename, activation, n_neurons=4, n_layers=2, learning_rates=[0.1, 0.01, 0.001, 0.0001], batch_size=30,
               mu=1, epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 12))
    title = f"Activation: {activation.__name__}  -   Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for lr, axis in zip(learning_rates, [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size, mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(activation, n_neurons))
        model.add(Output(Identity, n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        end_time = time.time()
        time_diff = round(end_time - start_time, 2)

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"Learning Rate: {lr}\nAverage Cost: {cost} - Time: {time_diff}s"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def compare_mu(filename, activation, n_neurons=4, n_layers=2, lr=0.001, batch_size=30, mus=[0.1, 0.01, 0.001, 0.001],
               epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 12))
    title = f"Activation: {activation.__name__}  -  Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for mu, axis in zip(mus, [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size, mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(activation, n_neurons))
        model.add(Output(Identity, n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        end_time = time.time()
        time_diff = round(end_time - start_time, 2)

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"Mu: {mu}\nAverage Cost: {cost} - Time: {time_diff}s"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    encoder = OneHotEncoder()
    scaler = StandardScaler()

    X, y = load_boston(return_X_y=True)
    y = y[:, np.newaxis]

    X = scaler.fit_transform(X)

    plot_four("oneHidden")
    plot_four("twoHidden", n_layers=2)
    plot_four("threeHidden", n_layers=3)

    plot_four("oneHiddenMoreN", n_neurons=10)
    plot_four("twoHiddenMoreN", n_neurons=10, n_layers=2)
    plot_four("threeHiddenMoreN", n_neurons=10, n_layers=3)

    compare_lr("lrSigmoid", activation=Sigmoid)
    compare_lr("lrSigmoid", activation=Tanh)
    compare_lr("lrSigmoid", activation=ReLu)
    compare_lr("lrSigmoid", activation=LeakyReLu)

    compare_mu("muSigmoid", activation=Sigmoid)
    compare_mu("muTanH", activation=Tanh)
    compare_mu("muReLu", activation=ReLu)
    compare_mu("muLeaky", activation=LeakyReLu)
