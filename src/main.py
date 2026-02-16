import numpy as np
from dense import Dense
from activation import Activation
from network import Network
from train import train, predict
from activationFn import *
from loss import *
from visual import plot_surface

def main():
    """
    Build, train and evaluate an MLP.
    Variables:
        Network structure
        Activation function from activationFn.py (default: sigmoid)
        Loss function from loss.py (default: MSE)
        Epochs and learning_rate (default: 1000, 0.1)
    """

    # configure network's structure, choose {func}
    layers = [
        Dense(2, 3)
        , Activation(tanh, tanh_prime)
        , Dense(3, 1)
        , Activation(tanh, tanh_prime)
    ]

    net = Network(layers, mse, mse_prime)  # create network object, choose {loss}
    # training
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (-1, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (-1, 1, 1))
    error = train(network=net, train_data=X, true_data=Y, epochs=1000, learning_rate=0.1)

    # testing
    T = np.reshape([1, 0], (2, 1))
    pred = predict(network=net, test_data=T)  # predict test_data

    # confidence computation
    pred = (pred+1)/2  # this line only needed when using tanh
    result = 1 if pred >= 0.5 else 0
    confidence = max(pred, 1 - pred) * 100

    #visualize
    plot_surface(net)

    # print results
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Error: {error:.4f}")


if __name__ == "__main__":
    main()





