from sklearn.base import BaseEstimator
import numpy as np

from src.inverse import invert_matrix


class LinearRegressor(BaseEstimator):
    def __init__(self, y_intercept: bool):
        self._weights = False
        self._bias = y_intercept
        super().__init__()


    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape != y.shape:
            raise ValueError("Shape of x and y do not match")
        normal = np.matmul(x.T, x)
        # add checks for failures
        inv_normal = invert_matrix(normal)
        moment = np.matmul(x.T, y)
        self._weights = np.matmul(inv_normal, moment)

        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._weights:
            raise ValueError("Model has not been fit yet, run .fit() first")
        n, p = x.shape
        if p != self._weights.shape[0]:
            raise ValueError(f"Model has been trained on data with feature size {self._weights[0]}")
        weights = np.repeat(self._weights, axis=-1, repeats=n)
        y_bar = np.dot(x, weights)
        if self._bias:
            y_bar += self._bias
        return y_bar

