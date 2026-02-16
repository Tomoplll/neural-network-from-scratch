def train(network, train_data, true_data, epochs=1000, learning_rate=0.1):
    """
    Train the network by performing forward and backward passes for each epoch.
    Updates layer parameters using training data to reduce prediction error.

    :param network: Network object
    :param train_data: Inputs for the first layer's values
    :param true_data: Corresponding expected outputs
    :param epochs: Number of training iterations
    :param learning_rate: Step size for updating parameters
    :return: Error of the last epoch
    """
    error = 0
    for epoch in range(epochs):
        error = 0
        for training_set, y_true in zip(train_data, true_data):
            y_pred = network.forward(training_set)  # forward-pass
            error += network.compute_loss(y_true, y_pred)  # compute error
            network.backward(y_true, y_pred, learning_rate)  # backward-pass
    return error / len(train_data)

def predict(network, test_data):
    """Perform forward pass and return the network's output for the given input."""
    return network.forward(test_data).item()
