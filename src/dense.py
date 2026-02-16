from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        Initialize the layer with random weights and biases, sets inputs to None
        :param input_size: Number of neurons in the layer (input)
        :param output_size: Number of neurons in the next layer (output)
        """
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, layer_input):
        self.input = layer_input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
