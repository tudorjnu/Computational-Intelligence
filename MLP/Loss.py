import numpy as np


class MeanSquaredError:

    def __init__(self):
        self.y_pred = 0
        self.y_true = 0

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        mse = (y_pred - y_true) ** 2
        if mse.shape[1] > 0:
            mse = np.mean(mse, axis=1, keepdims=True)
        return np.mean(mse)

    def backward(self):
        error = 2 * (self.y_pred - self.y_true)
        if self.y_pred.shape[1] > 0:
            error = np.mean(error, axis=1, keepdims=True)
        return error
