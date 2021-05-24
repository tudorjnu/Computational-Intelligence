import time

from deap import base
from deap import creator
from deap import tools
from deap.algorithms import eaSimple

toolbox = base.Toolbox()

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("seaborn-white")

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from MLP.Model import Sequential
from MLP.Activations import ReLu, Identity, Sigmoid
from MLP.Loss import MeanSquaredError
from MLP.Layers import Dense, Output

# DATASET
scaler = StandardScaler()
X, y = load_boston(return_X_y=True)
X = scaler.fit_transform(X)
y = y[:, np.newaxis]

# creating the network
model = Sequential()
model.add(Dense(Sigmoid(), 4))
model.add(Dense(Sigmoid(), 4))
model.add(Output(Identity(), 1, MeanSquaredError))
# initialise the weights
model.forward(X, y)


# ENCODING / getting the parameters space from the network
def get_parameters(model):
    parameters_counter = 0
    parameters = []
    for layer in model.layers:
        # count the biases parameters
        bias = np.count_nonzero(layer.bias)
        # count the weights parameter
        weights_param = np.count_nonzero(layer.weights.reshape((1, -1)))
        # count the layer parameters
        n_parameters = weights_param + bias
        # add the layer parameters number to the counter
        parameters_counter += n_parameters
        # add the details to the parameters list
        parameters.append((layer.weights.shape, bias, n_parameters))
    # return the parameters details, and the parameters counter
    return parameters, parameters_counter


network_parameters, num_parameters = get_parameters(model=model)

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# GENETIC HYPERPARAMETERS
HIGH_BOUND = 15.0
LOW_BOUND = -15.0
POPULATION_SIZE = 100
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5  # probability for mutation
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# define the fitness objective (minimise)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# define the individual
creator.create("Individual", list, fitness=creator.FitnessMin)


def random_float(min_boun, max_bound):
    return random.uniform(min_boun, max_bound)


# register floating attribute
toolbox.register("attr_float", random_float, LOW_BOUND, HIGH_BOUND)
# register individual creator function
toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=num_parameters)

# register the population creator function
toolbox.register("populationCreator", tools.initRepeat, list,
                 toolbox.individualCreator)


# create the fitness function
def evaluate_network(individual, network_parameters=network_parameters,
                     model=model, X=X, y=y):
    # create a list of the layers to be extracted
    layers = []
    # define the separator
    prev_separator = 0
    # use the parameters shape
    for parameters in network_parameters:
        # define the layer separator
        separator = parameters[2]
        # get the layer
        layer = individual[prev_separator:separator]
        # check if the layer has a bias
        if parameters[1] > 0:
            # get the bias
            bias = layer[0:parameters[1]]
            # get the weights
            weight = layer[parameters[1]:]
        else:
            # get the weights
            weight = layer
        # transform the weights from list to matrix
        weight = np.array(weight).reshape(parameters[0])
        # transform the bias into an array
        bias = np.array(bias)[np.newaxis,]
        # add the bias, and the weights to the layers list as a tuple
        layers.append((bias, weight))

    # for each layer parameters and the model layer
    for ga_layer, layer in zip(layers, model.layers):
        # make the model weights equal to the parameters
        layer.weights = ga_layer[1]
        # make the bias equal to the parameters
        layer.bias = ga_layer[0]

    # do a forward pass with the new weights and biases
    model.forward(X, y)
    # compute the cost
    cost = model.cost[-1]

    # return the cost
    return cost,


# add the evaluation function to the toolbox
toolbox.register("evaluate", evaluate_network)
# chose the selection method
toolbox.register("select", tools.selTournament, tournsize=2)
# chose the crossover method
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW_BOUND,
                 up=HIGH_BOUND, eta=CROWDING_FACTOR)
# chose the mutation method
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW_BOUND,
                 up=HIGH_BOUND, eta=CROWDING_FACTOR,
                 indpb=1.0 / len(network_parameters))


# Simple Genetic Algorithm
def simple(population_size=POPULATION_SIZE, max_generations=MAX_GENERATIONS,
           p_crossover=P_CROSSOVER, p_mutation=P_MUTATION,
           hof_size=HALL_OF_FAME_SIZE, verbose=False):
    # starting time of the algorithm
    start_time = time.time()
    # create initial population
    population = toolbox.populationCreator(n=population_size)

    # prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object
    hof = tools.HallOfFame(hof_size)

    # perform the GA
    population, logbook = eaSimple(population, toolbox, cxpb=p_crossover,
                                   mutpb=p_mutation,
                                   ngen=max_generations, stats=stats,
                                   halloffame=hof, verbose=verbose)

    # get the best individual
    best = hof.items[0]
    # get the best cost
    best_cost = best.fitness.values[0]
    # extract statistics
    min_cost, mean_cost = logbook.select("min", "avg")
    # get the time
    time_diff = time.time() - start_time

    # return the ga details
    return min_cost, mean_cost, time_diff, best_cost


def compare_mutation(filename, parameters=[]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(9, 9))

    for ax, parameter in zip([ax1, ax2, ax3, ax4], parameters):
        min, mean, time_diff, best_cost = simple(p_mutation=parameter)
        title = f"Mutation Probability: {parameter}"
        title += f"\n Min Cost: {best_cost:.2f} ({time_diff:.0f}s)"
        ax.plot(min, c="r", label="Min Cost")
        ax.plot(mean, c="g", label="Average Cost")
        ax.set_xlabel("Generation")
        ax.set_title(title)
        ax.set_ylabel("Cost")
        plt.ylim(ymax=500)
        plt.legend()
        plt.savefig(filename + ".png")
    plt.show()


def compare_crossover(filename, parameters=[]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(9, 9))

    for ax, parameter in zip([ax1, ax2, ax3, ax4], parameters):
        min, mean, time_diff, best_cost = simple(p_crossover=parameter)
        title = f"Crossover Probability: {parameter}"
        title += f"\n Min Cost: {best_cost:.2f} ({time_diff:.0f}s)"
        ax.plot(min, c="r", label="Min Cost")
        ax.plot(mean, c="g", label="Average Cost")
        ax.set_xlabel("Generation")
        ax.set_title(title)
        ax.set_ylabel("Cost")
        plt.ylim(ymax=500)
        plt.legend()
        plt.savefig(filename + ".png")

    plt.show()


def compare_population(filename, parameters=[]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(9, 9))

    for ax, parameter in zip([ax1, ax2, ax3, ax4], parameters):
        min, mean, time_diff, best_cost = simple(
            population_size=parameter)
        title = f"Population: {parameter}"
        title += f"\n Min Cost: {best_cost:.2f} ({time_diff:.0f}s)"
        ax.plot(min, c="r", label="Min Cost")
        ax.plot(mean, c="g", label="Average Cost")
        ax.set_xlabel("Generation")
        ax.set_title(title)
        ax.set_ylabel("Cost")
        plt.ylim(ymax=500)
        plt.legend()
        plt.savefig(filename + ".png")

    plt.show()


def compare_generations(filename, parameters=[]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(9, 9))

    for ax, parameter in zip([ax1, ax2, ax3, ax4], parameters):
        min, mean, time_diff, best_cost = simple(
            max_generations=parameter)
        title = f"Generations: {parameter}"
        title += f"\n Min Cost: {best_cost:.2f} ({time_diff:.0f}s)"
        ax.plot(min, c="r", label="Min Cost")
        ax.plot(mean, c="g", label="Average Cost")
        ax.set_xlabel("Generation")
        ax.set_title(title)
        ax.set_ylabel("Cost")
        plt.ylim(ymax=500)
        plt.legend()
        plt.savefig(filename + ".png")

    plt.show()


if __name__ == "__main__":
    compare_mutation("Graphs/GA/mutationComparison", parameters=[0.1, 0.4, 0.6, 0.9])
    compare_crossover("Graphs/GA/crossoverComparison", parameters=[0.1, 0.4, 0.6, 0.9])
    compare_population("Graphs/GA/populationComparison", parameters=[50, 100, 200, 400])
    compare_generations("Graphs/GA/generationsComparison", parameters=[50, 100, 200, 500])
