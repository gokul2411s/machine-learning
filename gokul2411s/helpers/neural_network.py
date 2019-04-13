import gokul2411s.helpers.math as mymath
import numpy as np

class NeuralNetwork(object):

    @staticmethod
    def builder():
        return NeuralNetwork.NeuralNetworkBuilder()

    def __init__(self, builder):
        self.input_dimensionality = builder.input_dimensionality
        self.layers = builder.layers
        self.activation_function = np.vectorize(builder.activation_function)
        self.activation_function_derivative = np.vectorize(builder.activation_function_derivative)
       
    def _validate_training_data(self, X, Y):
        if len(X) == 0:
            raise ValueError('No training data inputs.')
        if len(Y) == 0:
            raise ValueError('No training data outputs.')

        if  len(X) != len(Y):
            raise ValueError('Training data input and output have different number of data points.')

        dimensionality_X = len(X[0])
        if dimensionality_X != self.input_dimensionality:
            raise ValueError('Number of dimensions in training data input {} does not match expected {}.'.format(
                dimensionality_X, self.input_dimensionality))

        dimensionality_Y = len(Y[0])
        if dimensionality_Y != len(self.layers[-1][0]):
            raise ValueError('Number of dimensions in training data output {} does not match expected {}'.format(
                dimensionality_Y, len(self.layers[-1][0])))

    def _feed_forward(self, X):
        artificial_column = [[1] for _ in range(len(X))]

        layer_activation_outputs = [X] # sentinel, treating input as output from a zombie -1'th layer.
        derivatives_of_layer_activation_outputs_wrt_weighted_outputs = [None]
        for layer in self.layers:
            # Dims = (number of examples, number of nodes in previous layer).
            layer_input = layer_activation_outputs[-1]

            # Dims = (number of examples, number of nodes in previous layer + 1).
            layer_input_with_artificial_column = np.concatenate(
                    [artificial_column, layer_input], axis = 1)

            # Dims = (number of examples, number of nodes in current layer).
            layer_weighted_output = np.dot(layer_input_with_artificial_column, layer)

            # Dims = (number of examples, number of nodes in current layer).
            layer_activation_output = self.activation_function(layer_weighted_output)
            layer_activation_outputs.append(layer_activation_output)

            # Dims = (number of examples, number of nodes in current layer).
            derivative_of_layer_activation_output_wrt_weighted_output = self.activation_function_derivative(
                    layer_weighted_output)
            derivatives_of_layer_activation_outputs_wrt_weighted_outputs.append(
                    derivative_of_layer_activation_output_wrt_weighted_output)
        return (layer_activation_outputs, derivatives_of_layer_activation_outputs_wrt_weighted_outputs)

    def _feed_back(self, X, Y, learning_rate,
            layer_activation_outputs,
            derivatives_of_layer_activation_outputs_wrt_weighted_outputs):
        artificial_column = [[1] for _ in range(len(X))]

        # Compute error derivative assuming squared sum error function.
        # Dims = (number of examples, number of outputs from current layer). Current layer is the last layer now.
        derivative_of_error_wrt_layer_activation_output = np.subtract(layer_activation_outputs[-1], Y)
        for j in range(len(layer_activation_outputs) - 1, 0, -1):
            # Dims = (number of outputs from previous layer + 1, number of examples).
            derivative_of_layer_weighted_output_wrt_weight = np.concatenate(
                    [
                        artificial_column,
                        layer_activation_outputs[j - 1]
                    ],
                    axis = 1).T

            # Dims = (number of examples, number of outputs from current layer)
            derivative_of_error_wrt_layer_weighted_output =\
                    derivative_of_error_wrt_layer_activation_output\
                            * derivatives_of_layer_activation_outputs_wrt_weighted_outputs[j]

            # Dims = (number of outputs from previous layer + 1, number of outputs from current layer).
            derivative_of_error_wrt_weight = np.dot(
                    derivative_of_layer_weighted_output_wrt_weight,
                    derivative_of_error_wrt_layer_weighted_output)

            # Dims = (number of outputs from current layer, number of outputs from previous layer)
            # We don't care about derivative wrt the artificial 0'th column.
            derivative_of_layer_weighted_output_wrt_previous_layers_activation_output = np.delete(
                    self.layers[j - 1].T, 0, axis = 1)

            # Dims = (number of examples, number of outputs from previous layer).
            derivative_of_error_wrt_layer_activation_output = np.dot(
                    derivative_of_error_wrt_layer_weighted_output,
                    derivative_of_layer_weighted_output_wrt_previous_layers_activation_output)
           
            self.layers[j - 1] = np.subtract(
                    self.layers[j - 1],
                    (learning_rate * derivative_of_error_wrt_weight))

    def compute(self, X):
        """Computes the output of the neural network for a given set of inputs.

        Parameters:
        X (2D array like): input feature vectors, where each row corresponds to an input.

        Returns:
        2D array like: output labels, where each row corresponds to the input at the same row in X.

        """
        outputs, _ = self._feed_forward(X)
        return outputs[-1]

    def train(self, X, Y, learning_rate, num_iterations):
        """Trains the neural network based on the provided training data.

        Parameters:
        X (2D array like): training data feature vectors, where each row corresponds to an example.
        Y (2D array like): training data labels, where each row corresponds to an example.
        learning_rate (float): the rate at which to perform gradient descent.
        num_iterations (int): the number of times to repeat forward propagation and backward propagation.

        Returns:
        None: Updates the weights on the network's layers in place to fit the training data.

        """
        self._validate_training_data(X, Y)

        for _ in range(num_iterations):
            layer_activation_outputs, derivatives_of_layer_activation_outputs_wrt_weighted_outputs =\
                    self._feed_forward(X)
            self._feed_back(X, Y, learning_rate,\
                    layer_activation_outputs,
                    derivatives_of_layer_activation_outputs_wrt_weighted_outputs)

    class NeuralNetworkBuilder(object):

        def __init__(self):
            self.input_dimensionality = None
            self.num_nodes_in_prev_layer = None
            self.activation_function = mymath.sigmoid
            self.activation_function_derivative = mymath.sigmoid_derivative
            self.layers = []

        def with_input_dimensionality(self, dim):
            if self.input_dimensionality is not None:
                raise ValueError('Cannot set input dimensionality twice.')
            if not isinstance(dim, int):
                raise TypeError('Input dimensionality must be an integer.')
            if dim <= 0:
                raise ValueError('Input dimensionality must be a positive integer.')

            self.input_dimensionality = dim
            self.num_nodes_in_prev_layer = dim
            return self

        def with_new_layer(self, n, weights = None):
            """Creates a new layer, and wires it up automatically with the previous layer.

            Parameters:
            n (int): The number of nodes in the layer.
            weights (2D array like): An override useful for unit testing. Don't use otherwise.

            """
            if self.num_nodes_in_prev_layer is None:
                raise ValueError('Cannot create layers without setting input dimensionality first.')
            if not isinstance(n, int):
                raise TypeError('Number of nodes in layer must be an integer.')
            if n <= 0:
                raise ValueError('Number of nodes in layer must be a positive integer.')

            if weights is None:
                layer = np.random.rand(self.num_nodes_in_prev_layer + 1, n) # an extra weight for bias.
            else:
                layer = weights
            self.layers.append(layer)
            self.num_nodes_in_prev_layer = n
            return self
       
        def with_activation(self, func, derivative):
            """Set the activation function and its derivative.

            Parameters:
            func (callable): a function that accepts a float and returns float.
            derivative (callable): a function that corresponds to the derivative per differential calculus of
                                   the first argument func.

            """
            if not hasattr(func, '__call__'):
                raise TypeError('Activation function must be a function.')
            self.activation_function = func

            if not hasattr(derivative, '__call__'):
                raise TypeError('Activation function\'s derivative must be a function.')
            self.activation_function_derivative = derivative
            return self

        def build(self):
            if len(self.layers) == 0:
                raise ValueError('Network must have at least one layer.')
            return NeuralNetwork(self)
