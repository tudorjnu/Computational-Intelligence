import numpy as np
import math

from sklearn.datasets import load_boston, load_diabetes, make_regression
from Activations import Sigmoid, ReLu, Tanh, Identity, LeakyReLu
from Layers import Output, Dense
from Loss import MeanSquaredError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from Model import Sequential
import time

import matplotlib.pyplot as plt


def plot_four(filename, n_neurons=4, n_layers=1, lr=0.01, batch_size=30, mu=1,
              epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(12, 12))
    title = "Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for activation, axis in zip([Sigmoid, Tanh, ReLu, LeakyReLu],
                                [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size,
                           mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(activation(), n_neurons))
        model.add(
            Output(Identity(), n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        end_time = time.time()
        time_diff = end_time - start_time

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"{activation.__name__} \nAverage Cost: {cost} ({time_diff:.0f}s)"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def compare_lr(filename, activation, n_neurons=4, n_layers=2,
               learning_rates=[0.1, 0.01, 0.001, 0.0001], batch_size=30,
               mu=1, epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(12, 12))
    title = f"Activation: {activation.__name__}  -   Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for lr, axis in zip(learning_rates, [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size,
                           mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(activation(), n_neurons))
        model.add(
            Output(Identity(), n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        time_diff = time.time() - start_time

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"Learning Rate: {lr}\nAverage Cost: {cost} ({time_diff:.0f}s)"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def compare_mu(filename, activation, n_neurons=4, n_layers=2, lr=0.001,
               batch_size=30, mus=[0.1, 0.01, 0.001, 0.001],
               epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(12, 12))
    title = f"Activation: {activation.__name__}  -  Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for mu, axis in zip(mus, [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size,
                           mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(activation(), n_neurons))
        model.add(
            Output(Identity(), n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        time_diff = time.time() - start_time

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"Mu: {mu}\nAverage Cost: {cost} ({time_diff:.0f}s)"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def compare_sigmoid(filename, n_neurons=4, n_layers=1, lr=0.01, batch_size=30,
                    mu=1, epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(12, 12))
    title = "Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for alpha, axis in zip([100, 50, 30, 1], [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size,
                           mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(Sigmoid(alpha=alpha), n_neurons))
        model.add(
            Output(Identity(), n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        time_diff = time.time() - start_time

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"Alpha: {alpha} \nAverage Cost: {cost} ({time_diff}s)"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def compare_tanh_alpha(filename, n_neurons=4, n_layers=1, lr=0.01,
                       batch_size=30, mu=1, epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(12, 12))
    title = "Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for alpha, axis in zip([100, 50, 30, 1], [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size,
                           mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(Tanh(alpha=alpha), n_neurons))
        model.add(
            Output(Identity(), n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        time_diff = time.time() - start_time

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"Alpha: {alpha} \nAverage Cost: {cost} ({time_diff:.0f}s)"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def compare_tanh_beta(filename, n_neurons=4, n_layers=1, lr=0.01, batch_size=30,
                      mu=1, epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(12, 12))
    title = "Architecture: "
    for layer in range(n_layers):
        title += f"{n_neurons}-"
    title += "1"
    fig.suptitle(title, size=16)

    for beta, axis in zip([-100, -50, -30, -1], [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size,
                           mu=mu, seed=seed)
        for layer in range(n_layers):
            model.add(Dense(Tanh(beta=beta), n_neurons))
        model.add(
            Output(Identity(), n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        time_diff = time.time() - start_time

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"Beta: {beta} \nAverage Cost: {cost} ({time_diff:.0f}s)"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([-5, 500])
    plt.legend()
    plt.savefig(filename)
    plt.show()


def plot_four_diabetes(filename, n_neurons=4, n_layers=1, batch_size=30,
                       mu=1,
                       epochs=2000, seed=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(12, 12))

    no_layers = n_layers
    for activation, lr, axis in zip([Sigmoid, Tanh, ReLu, LeakyReLu],
                                    [0.001, 0.00001, 0.000001, 0.000001],
                                    [ax1, ax2, ax3, ax4]):
        model = Sequential(epochs=epochs, alpha=lr, batch_size=batch_size,
                           mu=mu, seed=seed)
        if activation.__name__ == "Sigmoid" or activation.__name__ == "Tanh":
            no_layers = n_layers - 1
        for layer in range(no_layers):
            model.add(Dense(activation(), n_neurons))
        model.add(
            Output(Identity(), n_neurons=1, cost_function=MeanSquaredError))
        start_time = time.time()
        model.fit(X, y, verbose=False)
        end_time = time.time()
        time_diff = end_time - start_time

        cost = model.averages[-1]
        if cost > 100000:
            cost = "> 100000"
        elif math.isnan(cost):
            cost = "inf"
        else:
            cost = round(cost, 2)
        subtitle = f"{activation.__name__} \nAverage Cost: {cost} ({time_diff:.0f}s)"
        axis.set_title(subtitle, size=16)
        axis.plot(model.cost, color='green', label="cost")
        axis.plot(model.averages, color='red', label="average cost")
        axis.set_xlabel('Epochs', size=10)
        axis.set_ylim([0, 10000])
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

    plot_four("MLP/Graphs/oneHidden")
    plot_four("MLP/Graphs/twoHidden", n_layers=2)
    plot_four("MLP/Graphs/threeHidden", n_layers=3)

    plot_four("MLP/Graphs/oneHiddenMoreN", n_neurons=10)
    plot_four("MLP/Graphs/twoHiddenMoreN", n_neurons=10, n_layers=2)

    compare_lr("MLP/Graphs/lrSigmoid", activation=Sigmoid)
    compare_lr("MLP/Graphs/lrTanH", activation=Tanh)
    compare_lr("MLP/Graphs/lrReLu", activation=ReLu)
    compare_lr("MLP/Graphs/lrLeakyReLu", activation=LeakyReLu)

    compare_mu("MLP/Graphs/muSigmoid", activation=Sigmoid)
    compare_mu("MLP/Graphs/muTanH", activation=Tanh)
    compare_mu("MLP/Graphs/muReLu", activation=ReLu)
    compare_mu("MLP/Graphs/muLeaky", activation=LeakyReLu)

    X, y = load_diabetes(return_X_y=True)
    y = y[:, np.newaxis]

    X = scaler.fit_transform(X)

    plot_four_diabetes("MLP/Graphs/diabetesComparison", epochs=40_000, n_neurons=40,
                       n_layers=3)
