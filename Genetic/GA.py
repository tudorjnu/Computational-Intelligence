from deap import base
from deap import creator
from deap import tools
from deap.algorithms import eaSimple

import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import elitism

from sklearn.datasets import load_boston

from MLP.Model import Sequential
from MLP.Activations import ReLu, Identity, Sigmoid
from MLP.Loss import MeanSquaredError
from MLP.Layers import Dense, Output

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# DATASET
X, y = load_boston(return_X_y=True)
y = y[:, np.newaxis]

# creating the network
model = Sequential()
model.add(Dense(ReLu(), 5))
model.add(Dense(ReLu(), 5))
model.add(Dense(ReLu(), 5))
model.add(Output(Identity(), 1, MeanSquaredError()))
# initialise the weights
model.forward(X, y)


# ENCODING / getting the parameters space from the network
def get_parameters(model):
    parameters_counter = 0
    parameters = []
    for layer in model.layers:
        bias = np.count_nonzero(layer.bias)
        weights_param = np.count_nonzero(layer.weights.reshape((1, -1)))
        n_parameters = weights_param + bias
        parameters_counter += n_parameters
        parameters.append((layer.weights.shape, bias, n_parameters))
    return parameters, parameters_counter


network_parameters, num_parameters = get_parameters(model=model)
print(network_parameters)

# GENETIC HYPERPARAMETERS
HIGH_BOUND = 5.0
LOW_BOUND = -5.0

# Genetic Algorithm constants:
POPULATION_SIZE = 100
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.8  # (try also 0.5) probability for mutating an individual
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation
ELITISM = True

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def random_float(min_boun, max_bound):
    return random.uniform(min_boun, max_bound)


toolbox.register("attr_float", random_float, LOW_BOUND, HIGH_BOUND)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_parameters)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

individual = toolbox.individualCreator()


def evaluate_network(individual, network_parameters=network_parameters, model=model, X=X, y=y):
    layers = []
    prev_separator = 0
    for parameters in network_parameters:
        separator = parameters[2]
        layer = individual[prev_separator:separator]
        if parameters[1] > 0:
            bias = layer[0:parameters[1]]
            weight = layer[parameters[1]:]
        else:
            weight = layer
        weight = np.array(weight).reshape(parameters[0])
        bias = np.array(bias)[np.newaxis,]
        layers.append((bias, weight))

    for ga_layer, layer in zip(layers, model.layers):
        layer.weights = ga_layer[1]
        layer.bias = ga_layer[0]

    model.forward(X, y)
    cost = model.cost[-1]

    return cost,


toolbox.register("evaluate", evaluate_network)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW_BOUND, up=HIGH_BOUND, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW_BOUND, up=HIGH_BOUND, eta=CROWDING_FACTOR,
                 indpb=1.0 / len(network_parameters))


# Genetic Algorithm flow:
def simple_with_elitism():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Fitness = ", best.fitness.values[0])

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations - Elitism')
    plt.ylim([-1000000, 40000000])
    plt.savefig("Elitism.png")
    plt.show()


# Genetic Algorithm flow:
def simple():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # perform the Genetic Algorithm flow:
    population, logbook = eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                   stats=stats, verbose=True)

    # Genetic Algorithm is done - extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average Fitness over Generations')
    plt.ylim([-1000000, 40000000])
    plt.savefig("Simple.png")
    plt.show()


if __name__ == "__main__":
    simple_with_elitism()
    simple()
