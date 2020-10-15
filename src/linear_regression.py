from sklearn.base import BaseEstimator
import numpy as np

from src.inverse import invert_matrix


class LinearRegressor(BaseEstimator):
    """"
    A linear regression algorithm to model the relationship between two variables by fitting a linear equation to observed data.
    Currently supports data being off-centered (does not normalise data).
    """

    def __init__(self, weights: np.ndarray = None):
        self._weights = weights
        self._fit = False
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape != y.shape:
            raise ValueError("Shape of x and y do not match")
        x = self._add_bias(x)
        normal = np.matmul(x.T, x)
        inv_normal = invert_matrix(normal)
        moment = np.matmul(x.T, y)
        self._weights = np.matmul(inv_normal, moment)
        # return score
        self._fit = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._fit:
            raise ValueError("Model has not been fit yet, run .fit() first")
        x = self._add_bias(x)
        n, p = x.shape
        if p != self._weights.shape[0]:
            raise ValueError(f"Model has been trained on data with feature size {self._weights.shape[0]} but data has "
                             f"feature size {p}")
        return np.dot(x, self._weights)

    def _add_bias(self, x: np.array):
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
