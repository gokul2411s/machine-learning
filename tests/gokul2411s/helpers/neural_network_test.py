import gokul2411s.helpers.neural_network as nn
import numpy as np
import unittest

class NeuralNetworkTest(unittest.TestCase):

    def test_builder_raises_under_invalid_input_dimensionality_specification(self):
        with self.assertRaises(ValueError) as cm:
            nn.NeuralNetwork.builder().with_new_layer(1).build()
        self.assertEqual(str(cm.exception), 'Cannot create layers without setting input dimensionality first.')

        with self.assertRaises(TypeError) as cm:
            nn.NeuralNetwork.builder().with_input_dimensionality(4.5)
        self.assertEqual(str(cm.exception), 'Input dimensionality must be an integer.')

        with self.assertRaises(ValueError) as cm:
            nn.NeuralNetwork.builder().with_input_dimensionality(-4)
        self.assertEqual(str(cm.exception), 'Input dimensionality must be a positive integer.')

    def test_builder_raises_under_invalid_layer_specification(self):
        with self.assertRaises(ValueError) as cm:
            nn.NeuralNetwork.builder().build()
        self.assertEqual(str(cm.exception), 'Network must have at least one layer.')

        with self.assertRaises(TypeError) as cm:
            nn.NeuralNetwork.builder().with_input_dimensionality(1).with_new_layer(4.5)
        self.assertEqual(str(cm.exception), 'Number of nodes in layer must be an integer.')

        with self.assertRaises(ValueError) as cm:
            nn.NeuralNetwork.builder().with_input_dimensionality(1).with_new_layer(-4)
        self.assertEqual(str(cm.exception), 'Number of nodes in layer must be a positive integer.')

    def test_builder_raises_under_invalid_activation_specification(self):
        with self.assertRaises(TypeError) as cm:
            nn.NeuralNetwork.builder().with_activation(1, lambda x: x)
        self.assertEqual(str(cm.exception), 'Activation function must be a function.')

        with self.assertRaises(TypeError) as cm:
            nn.NeuralNetwork.builder().with_activation(lambda x: x, 1)
        self.assertEqual(str(cm.exception), 'Activation function\'s derivative must be a function.')

    def test_builder_builds_correctly(self):
        network = nn.NeuralNetwork.builder()\
                .with_input_dimensionality(1)\
                .with_new_layer(3)\
                .with_new_layer(5)\
                .build()

        self.assertEqual(len(network.layers), 2)
        self.assertEqual(network.layers[0].shape, (2, 3))
        self.assertEqual(network.layers[1].shape, (4, 5))

    def test_training_raises_under_invalid_training_data_specification(self):
        network = nn.NeuralNetwork.builder()\
                .with_input_dimensionality(3)\
                .with_new_layer(2)\
                .build()

        with self.assertRaises(ValueError) as cm:
            network.train(X = [], Y = [], learning_rate = 0.1, num_iterations = 100)
        self.assertEqual(str(cm.exception), 'No training data inputs.')

        with self.assertRaises(ValueError) as cm:
            network.train(X = [[1]], Y = [], learning_rate = 0.1, num_iterations = 100)
        self.assertEqual(str(cm.exception), 'No training data outputs.')

        with self.assertRaises(ValueError) as cm:
            network.train(X = [[1], [2]], Y = [[1]], learning_rate = 0.1, num_iterations = 100)
        self.assertEqual(str(cm.exception), 'Training data input and output have different number of data points.')

        with self.assertRaises(ValueError) as cm:
            network.train(X = [[1, 2]], Y = [[1]], learning_rate = 0.1, num_iterations = 100)
        self.assertEqual(str(cm.exception), 'Number of dimensions in training data input 2 does not match expected 3.')

        with self.assertRaises(ValueError) as cm:
            network.train(X = [[1, 2, 3]], Y = [[1]], learning_rate = 0.1, num_iterations = 100)
        self.assertEqual(str(cm.exception), 'Number of dimensions in training data output 1 does not match expected 2')

    def test_training_with_single_training_data_point_runs_without_raising_errors(self):
        network = nn.NeuralNetwork.builder()\
                .with_input_dimensionality(3)\
                .with_new_layer(2)\
                .with_new_layer(5)\
                .build()

        network.train(
                X = 
                [
                    [0.1, 0.2, 0.3]
                ],
                Y =
                [
                    [1, 2, 3, 4, 5]
                ],
                learning_rate = 0.1,
                num_iterations = 1000)

    def test_training_with_multiple_training_data_points_runs_without_raising_errors(self):
        network = nn.NeuralNetwork.builder()\
                .with_input_dimensionality(3)\
                .with_new_layer(2)\
                .with_new_layer(4)\
                .with_new_layer(5)\
                .build()

        network.train(
                X =
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]
                ],
                Y =
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10]
                ],
                learning_rate = 0.1,
                num_iterations = 1000)

    def test_compute_works_as_expected(self):
        network = nn.NeuralNetwork.builder()\
                .with_input_dimensionality(1)\
                .with_new_layer(2, weights = [[1, 2], [3, 4]])\
                .with_activation(lambda x: x, lambda x: 1)\
                .build()

        self.assertEqual(network.compute([[1]]).tolist(), [[4, 6]])

