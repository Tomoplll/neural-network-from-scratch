import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross(y_true, y_pred):
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def binary_cross_prime(y_true, y_pred):  # can only be used if final activation is sigmoid
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (-y_true / p + (1 - y_true) / (1 - p)) / y_true.size

def categorical_cross(y_true, y_pred):
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(p), axis=1))

def categorical_cross_prime(y_true, y_pred):
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - (y_true / p) / y_true.shape[0]