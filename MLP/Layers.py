import numpy as np
from sklearn.datasets import load_boston
from MLP.Activations import Sigmoid, ReLu, Tanh, Identity
from MLP.Loss import MeanSquaredError
from sklearn.preprocessing import StandardScaler


class Layer:

    def __init__(self, activation, n_neurons):
        self.n_neurons = n_neurons
        self.activation = activation
        self.weights = None
        self.bias = None
        self.input = np.array([])
        self.z = np.array([])
        self.local_error = np.array([])
        self.weights_gradient = np.array([])
        self.bias_gradient = np.array([])

    def initialise_weights(self):
        self.weights = np.random.random((self.input.shape[1], self.n_neurons))
        self.bias = np.random.random((1, self.n_neurons))

    def update_weights(self, alpha=0.01, mu=1):
        self.weights_gradient = self.input.T.dot(self.local_error)
        self.bias_gradient = self.local_error
        self.weights = mu * self.weights - alpha * self.weights_gradient
        self.bias = mu * self.bias - alpha * self.bias_gradient


class Dense(Layer):

    def __init__(self, activation, n_neurons):
        super().__init__(activation, n_neurons)

    def forward(self, input):
        self.input = input
        if self.weights is None:
            self.initialise_weights()
        self.z = self.input.dot(self.weights) + self.bias
        self.a = self.activation.forward(self.z)
        return self.a

    def backpropagate_error(self, upper_error):
        self.local_error = upper_error * self.activation.backward()
        previous_a_error = self.local_error.dot(self.weights.T)
        return previous_a_error


class Output(Layer):

    def __init__(self, activation, n_neurons, cost_function):
        super().__init__(activation, n_neurons)
        self.cost = np.array([])
        self.cost_function = cost_function()
        self.y_true = np.array([])

    def initialise_weights(self):
        self.weights = np.random.random((self.input.shape[1], self.n_neurons))

    def update_weights(self, alpha=0.01, mu=1):
        self.weights_gradient = np.mean(self.input.T.dot(self.local_error), axis=0)
        self.weights = mu * self.weights - alpha * self.weights_gradient

    def forward(self, input):
        self.input = input
        if self.weights is None:
            self.initialise_weights()
        self.z = self.input.dot(self.weights)
        self.a = self.activation.forward(self.z)
        return self.a

    def calculate_cost(self, y_true):
        self.y_true = y_true
        self.cost = self.cost_function.forward(self.a, y_true)
        return self.cost

    def backpropagate_error(self):
        self.local_error = self.cost_function.backward() * self.activation.backward()
        previous_a_error = self.local_error.dot(self.weights.T)
        return previous_a_error


if __name__ == "__main__":
    scaler = StandardScaler()
    np.random.seed(42)

    X, y = load_boston(return_X_y=True)
    n_samples = 3
    X = X[:n_samples, :2]
    y = y[:n_samples, np.newaxis]
    y = scaler.fit_transform(y)
    X = scaler.fit_transform(X)

    input_layer = Input(activation=Identity, n_neurons=2)
    hidden_layer1 = Dense(activation=Sigmoid, n_neurons=3)
    hidden_layer2 = Dense(activation=Sigmoid, n_neurons=2)
    output_layer = Output(activation=Identity, n_neurons=1, cost_function=MeanSquaredError)

    print("\nInput:")
    print(X)
    input_layer_output = input_layer.forward(X)
    print("\nInput Layer Output: ")
    print(input_layer_output)
    print("\nHidden Layer 1 Output: ")
    hidden_layer1_output = hidden_layer1.forward(input_layer_output)
    print(hidden_layer1_output)
    print("\nHidden Layer 2 Output:")
    hidden_layer2_output = hidden_layer2.forward(hidden_layer1_output)
    print(hidden_layer2_output)

    output_layer_output = output_layer.forward(hidden_layer2_output)
    print("\nOutput Layer Output:")
    print(output_layer_output)
    print("\nY true")
    print(y)

    print("\nCost")
    cost = output_layer.calculate_cost(y_true=y)
    print(cost)

    print("\n\nBackpropagation\n")
    prev_error = output_layer.backpropagate_error()

    print("Output Layer Error")
    output_layer_error = output_layer.local_error
    output_layer_weights = output_layer.weights
    print(output_layer_error)

    print("\nHidden Layer 2 Error")
    prev_error = hidden_layer2.backpropagate_error(prev_error)
    print(hidden_layer2.local_error)

    print("\nHidden Layer 1 Error")
    hidden_layer1_error = hidden_layer1.backpropagate_error(prev_error)
    print(hidden_layer1.local_error)

    print("output Layer Bias")
    print(output_layer.bias)
    print("\nWeights Update")
    print("\nOriginal")
    print(output_layer.weights)
    output_layer.update_weights()
    print("\ngradient")
    print(output_layer.weights_gradient)
    print("\nNew Weights")
    print(output_layer.weights)
    print("\nBias:")
    print(output_layer.bias)
    print(output_layer.bias_gradient)

    print("\nHidden 2")
    print("\nWeights Update")
    print("\nOriginal")
    print(hidden_layer2.weights)
    print("\nBias:")
    print(hidden_layer2.bias)
    hidden_layer2.update_weights()
    print("\ngradient")
    print(hidden_layer2.weights_gradient)
    print("\nNew Weights")
    print(hidden_layer2.weights)

    print("Hidden 1")
    print("\nWeights")
    print(hidden_layer1.weights)
    hidden_layer1.update_weights()
    print("\ngradient")
    print(hidden_layer1.weights_gradient)
    print("\nNew Weights")
    print(hidden_layer1.weights)

    print("\nInput:")
    print(X)
    input_layer_output = input_layer.forward(X)
    print("\nInput Layer Output: ")
    print(input_layer_output)
    print("\nHidden Layer 1 Output: ")
    hidden_layer1_output = hidden_layer1.forward(input_layer_output)
    print(hidden_layer1_output)
    print("\nHidden Layer 2 Output:")
    hidden_layer2_output = hidden_layer2.forward(hidden_layer1_output)
    print(hidden_layer2_output)
    print("\nOutput Layer Output:")
    output_layer_output = output_layer.forward(hidden_layer2_output)
    print(output_layer_output)

    print("\nCost")
    cost = output_layer.calculate_cost(y_true=y)
    print(cost)

    print("\n\nBackpropagation\n")
    prev_error = output_layer.backpropagate_error()

    print("Output Layer Error")
    output_layer_error = output_layer.local_error
    output_layer_weights = output_layer.weights
    print(output_layer_error)

    print("\nHidden Layer 2 Error")
    prev_error = hidden_layer2.backpropagate_error(prev_error)
    print(hidden_layer2.local_error)

    print("\nHidden Layer 1 Error")
    hidden_layer1_error = hidden_layer1.backpropagate_error(prev_error)
    print(hidden_layer1.local_error)

    print("output Layer Bias")
    print(output_layer.bias)
    print("\nWeights Update")
    print("\nOriginal")
    print(output_layer.weights)
    output_layer.update_weights()
    print("\ngradient")
    print(output_layer.weights_gradient)
    print("\nNew Weights")
    print(output_layer.weights)
    print("\nBias:")
    print(output_layer.bias)
    print(output_layer.bias_gradient)

    print("\nHidden 2")
    print("\nWeights Update")
    print("\nOriginal")
    print(hidden_layer2.weights)
    print("\nBias:")
    print(hidden_layer2.bias)
    hidden_layer2.update_weights()
    print("\ngradient")
    print(hidden_layer2.weights_gradient)
    print("\nNew Weights")
    print(hidden_layer2.weights)

    print("Hidden 1")
    print("\nWeights")
    print(hidden_layer1.weights)
    hidden_layer1.update_weights()
    print("\ngradient")
    print(hidden_layer1.weights_gradient)
    print("\nNew Weights")
    print(hidden_layer1.weights)
