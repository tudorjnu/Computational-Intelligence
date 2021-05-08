import numpy as np

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