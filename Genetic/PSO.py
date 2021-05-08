from deap import base
from deap import creator
from deap import tools

import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from MLP.Model import Sequential
from MLP.Activations import ReLu, Identity, Sigmoid
from MLP.Loss import MeanSquaredError
from MLP.Layers import Dense, Output

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# DATASET
scaler = StandardScaler()
X, y = load_boston(return_X_y=True)
X = scaler.fit_transform(X)
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


def random_float(min_bound, max_bound):
    return random.uniform(min_bound, max_bound)


def create_network(individual, network_parameters=network_parameters):
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

        return layers


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


# constants:
POPULATION_SIZE = 35
MAX_GENERATIONS = 20000
DIMENSIONS = num_parameters
MIN_START_POSITION, MAX_START_POSITION = -10.0, 10.0
MIN_SPEED, MAX_SPEED = -0.04, 0.04
MAX_LOCAL_UPDATE_FACTOR = MAX_GLOBAL_UPDATE_FACTOR = 2.0

# set the random seed:
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# define the particle class based on ndarray:
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, best=None)


# create and initialize a new particle:
def createParticle():
    particle = creator.Particle(np.random.uniform(MIN_START_POSITION,
                                                  MAX_START_POSITION,
                                                  DIMENSIONS))
    particle.speed = np.random.uniform(MIN_SPEED, MAX_SPEED, DIMENSIONS)
    return particle


# create the 'particleCreator' operator to fill up a particle instance:
toolbox.register("particleCreator", createParticle)

# create the 'population' operator to generate a list of particles:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.particleCreator)


def updateParticle(particle, best):
    # create random factors:
    localUpdateFactor = np.random.uniform(0, MAX_LOCAL_UPDATE_FACTOR, particle.size)
    globalUpdateFactor = np.random.uniform(0, MAX_GLOBAL_UPDATE_FACTOR, particle.size)

    # calculate local and global speed updates:
    localSpeedUpdate = localUpdateFactor * (particle.best - particle)
    globalSpeedUpdate = globalUpdateFactor * (best - particle)

    # scalculate updated speed:
    particle.speed = particle.speed + (localSpeedUpdate + globalSpeedUpdate)

    # enforce limits on the updated speed:
    particle.speed = np.clip(particle.speed, MIN_SPEED, MAX_SPEED)

    # replace particle position with old-position + speed:
    particle[:] = particle + particle.speed


toolbox.register("update", updateParticle)

toolbox.register("evaluate", evaluate_network)


def main():
    # create the population of particle population:
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    for generation in range(MAX_GENERATIONS):

        # evaluate all particles in polulation:
        for particle in population:

            # find the fitness of the particle:
            particle.fitness.values = toolbox.evaluate(particle)

            # particle best needs to be updated:
            if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values

            # global best needs to be updated:
            if best is None or best.size == 0 or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values

        # update each particle's speed and position:
        for particle in population:
            toolbox.update(particle, best)

        # record the statistics for the current generation and print it:
        logbook.record(gen=generation, evals=len(population), **stats.compile(population))
        print(logbook.stream)

    # print info for best solution found:
    print("-- Best Fitness = ", best.fitness.values[0])


if __name__ == "__main__":
    main()
