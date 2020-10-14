from sklearn.base import BaseEstimator
import numpy as np

from src.inverse import invert_matrix


class LinearRegressor(BaseEstimator):
    def __init__(self, y_intercept: bool):
        self.weights = False
        self.bias = y_intercept

    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape != y.shape:
            raise ValueError("Shape of x and y do not match")

        normal = np.matmul(x.T, x)

        # add checks for failures
        inv_normal = invert_matrix(normal)

        moment = np.matmul(x.T, y)
        self.weights = np.matmul(inv_normal, moment)

        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.weights:
            raise ValueError("Model has not been fit yet, run .fit() first")
        n, p = x.shape
        weights = np.repeat(self.weights, axis=-1, repeats=n)
        y_bar = np.dot(weights, x)
        if self.bias:
            y_bar += self.bias
        return y_bar

