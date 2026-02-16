from layer import Layer
from activationFn import sigmoid, sigmoid_prime
import numpy as np

class Activation(Layer):
    def __init__(self, func=sigmoid, func_prime=sigmoid_prime):
        """
        Initialize the activation layer with {func} as the activation function
        to include non-linearity in the model.
        :param func: Chosen activation function from activationFn.py
        :param func_prime: Corresponding prime function
        """
        super().__init__()
        self.activation_fn = func
        self.activation_fn_prime = func_prime

    def forward(self, layer_input):
        self.input = layer_input
        return self.activation_fn(self.input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_fn_prime(self.input)
