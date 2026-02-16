from loss import mse, mse_prime

class Network:
    def __init__(self, layers, loss=mse, loss_prime=mse_prime):
        """
        Initialize the network with the chosen structure and loss function
        :param layers: Layers list of the network
        :param loss: Chosen loss function from loss.py
        :param loss_prime: Corresponding prime function
        """
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime

    def forward(self, input_data):
        """Perform forward pass through all the layers given single data-set"""
        x = input_data
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true, y_pred, learning_rate):
        """Perform backpropagation through all layers and update the weights"""
        gradient = self.loss_prime(y_true, y_pred)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def compute_loss(self, y_true, y_pred):
        """Compute the error using {loss} func"""
        return self.loss(y_true, y_pred)



