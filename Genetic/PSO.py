import time

import matplotlib.style
from deap import base
from deap import creator
from deap import tools

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
model.add(Dense(Sigmoid(), 5))
model.add(Dense(Sigmoid(), 5))
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


# getting the parameters details
network_parameters, num_parameters = get_parameters(model=model)


# create the network from genotype
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
        # add the bias and the weights to the layers list as a tuple
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


# HYPERPARAMETERS
SWARM_SIZE = 35
MAX_GENERATIONS = 200
MIN_START_POSITION, MAX_START_POSITION = -10.0, 10.0  # starting position
MIN_SPEED, MAX_SPEED = -04., 0.4  # maximum speed of the particles
MAX_LOCAL_UPDATE_FACTOR = MAX_GLOBAL_UPDATE_FACTOR = 2.0

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None,
               best=None)


# create and initialize a new particle:
def createParticle():
    # create the particle
    particle = creator.Particle(np.random.uniform(MIN_START_POSITION,
                                                  MAX_START_POSITION,
                                                  num_parameters))
    # create the particle speed
    particle.speed = np.random.uniform(MIN_SPEED, MAX_SPEED, num_parameters)
    return particle


# assign the function to the particle creator
toolbox.register("particleCreator", createParticle)

# create the population
toolbox.register("populationCreator", tools.initRepeat, list,
                 toolbox.particleCreator)


# define the update function for the particle
def updateParticle(particle, best):
    # create random factors
    localUpdateFactor = np.random.uniform(0, MAX_LOCAL_UPDATE_FACTOR,
                                          particle.size)
    globalUpdateFactor = np.random.uniform(0, MAX_GLOBAL_UPDATE_FACTOR,
                                           particle.size)

    # calculate local and global speed updates
    localSpeedUpdate = localUpdateFactor * (particle.best - particle)
    globalSpeedUpdate = globalUpdateFactor * (best - particle)

    # calculate updated speed:
    particle.speed = particle.speed + (localSpeedUpdate + globalSpeedUpdate)

    # enforce limits on the updated speed
    particle.speed = np.clip(particle.speed, MIN_SPEED, MAX_SPEED)

    # replace particle position with old-position + speed
    particle[:] = particle + particle.speed


# register the updating function
toolbox.register("update", updateParticle)

# register the evaluation function
toolbox.register("evaluate", evaluate_network)


# crete the function that runs the algorithm
def run_pso(population_size=SWARM_SIZE, max_generations=MAX_GENERATIONS,
            ):
    # get the start time
    start_time = time.time()
    # create the population
    population = toolbox.populationCreator(n=population_size)

    # prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # create the logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    # for each generation
    for generation in range(max_generations):

        # evaluate all particles in population
        for particle in population:

            # find the fitness of the particle
            particle.fitness.values = toolbox.evaluate(particle)

            # update the particle best
            if particle.best is None or particle.best.size == 0 or \
                    particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values

            # update the global best
            if best is None or best.size == 0 or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values

        # update each particle's speed and position
        for particle in population:
            toolbox.update(particle, best)

        # record the statistics
        logbook.record(gen=generation, evals=len(population),
                       **stats.compile(population))

    # get the statistics
    min_cost, mean_cost = logbook.select("min", "avg")
    # compute the time
    time_diff = time.time() - start_time
    # return the details
    return min_cost, mean_cost, time_diff, best.fitness.values[0]


def compare_swarm(filename, parameters=[]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(9, 9))

    for ax, parameter in zip([ax1, ax2, ax3, ax4], parameters):
        min, mean, time_diff, best_cost = run_pso(population_size=parameter)
        title = f"Swarm Size: {parameter}"
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


def compare_generation(filename, parameters=[]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,
                                                 figsize=(9, 9))

    for ax, parameter in zip([ax1, ax2, ax3, ax4], parameters):
        min, mean, time_diff, best_cost = run_pso(max_generations=parameter)
        title = f"Generations: {parameter}"
        title += f"\n Min Cost: {best_cost:.2f} ({time_diff:.0f}s)"
        ax.plot(min, c="r", label="Min Cost")
        ax.plot(mean, c="g", label="Average Cost")
        ax.set_xlabel("Generation")
        ax.set_title(title)
        ax.set_ylabel("Cost")
        plt.ylim([0, 500])
    plt.legend()
    plt.savefig(filename + ".png")
    plt.show()


if __name__ == "__main__":
    compare_swarm("Graphs/PSO/swarmSize",
                  parameters=[20, 50, 100, num_parameters * 5])
    # compare_generation("Graphs/PSO/swarmGenerations", parameters=[50, 100, 200, 500])
