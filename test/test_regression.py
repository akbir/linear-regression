from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression as Sklearn_Reg

from src.linear_regression import LinearRegressor

rng = np.random.default_rng(42)
def _create_linear_values(num_samples, feature_size, coefficient, bias):
    x = rng.random((num_samples, feature_size))
    y = coefficient * x + bias
    return x, y

@pytest.mark.integration
class TestIntegrationsLinearRegression:
    @pytest.mark.parametrize('num_samples,features,coefficient,bias',
                             [(20, 1, 5, 0),
                              (100, 25, 0.1, 0),
                              (20, 1, 5, 10),
                              (3, 1, 5, 10),
                              ]
                             )
    def test_linear_regression(self, num_samples, features, coefficient, bias):
        x, y = _create_linear_values(num_samples, features, coefficient, bias)
        model = LinearRegressor()
        model.fit(x, y)
        y_bar = model.predict(x)
        np.testing.assert_almost_equal(y_bar, y, decimal=6)

    @pytest.mark.integration
    @pytest.mark.parametrize('func',
                             [lambda x: x * x, lambda x: 2 * x + 5, lambda x: np.linalg.norm(x, axis=1)]
                             )
    def test_parity_with_sklearn(self, func):
        num_samples = 100
        x = np.random.rand(2 * num_samples, 20)
        x_train = x[:num_samples]
        y = func(x_train)
        y = np.reshape(y, (num_samples, -1))

        m1 = LinearRegressor()
        m2 = Sklearn_Reg()

        m1.fit(x_train, y)
        m2.fit(x_train, y)

        p1 = m1.predict(x[num_samples:])
        p2 = m2.predict(x[num_samples:])

        np.testing.assert_almost_equal(p1, p2, decimal=6)


class TestsLinearRegression:
    @pytest.mark.parametrize('num_samples,features,coeff,bias',
                             [(20, 10, 3, 0),
                              (100, 25, 5, 0),
                              (20, 1, 7, 10),
                              (3, 1, 5, 0.5),
                              ]
                             )

    @mock.patch("src.linear_regression.invert_matrix", autospec=True)
    def test_regression_fit(self, mock_invert, num_samples, features, coeff, bias):
        # mock out our custom matrix inverse
        mock_invert.side_effect = lambda z: np.linalg.inv(z)
        x, y = _create_linear_values(num_samples, features, coeff, bias)
        model = LinearRegressor()
        model.set_params = Mock()
        model.fit(x, y)

        mock_invert.assert_called_once()
        model.set_params.assert_called_once()

        # fetch call argument
        weights = model.set_params.call_args[1]['weights']

        # assert weight called correctly
        np.testing.assert_almost_equal(weights[1:], coeff * np.eye(features, features))
        np.testing.assert_almost_equal(weights[0], bias * np.ones((features,)))


    @pytest.mark.parametrize('num_samples,features,coeff,bias',
                             [(20, 10, 3, 0),
                              (100, 25, 5, 0),
                              (20, 1, 7, 10),
                              (3, 3, 5, 0.5),
                              ]
                             )
    def test_regression_predict(self, num_samples, features, coeff, bias):
        x, y = _create_linear_values(num_samples, features, coeff, bias)
        model = LinearRegressor()

        # mock fit
        model._fit = True
        mock_weights = np.concatenate([bias * np.ones((1, features)),coeff * np.eye(features, features)], axis=0)
        model.set_params(weights=mock_weights)

        y_bar = model.predict(x)
        np.testing.assert_almost_equal(y, y_bar)

    def test_predict_before_train_error(self):
        model = LinearRegressor()
        with pytest.raises(ValueError) as e:
            model.predict(np.zeros((10, 2)))
        assert str(e.value) == 'Model has not been fit yet, run .fit() first'

    def test_non_matching_num_samples_error(self):
        model = LinearRegressor()
        with pytest.raises(ValueError) as e:
            model.fit(np.ones((5, 6)), np.ones((10,6)))
        assert str(e.value) == 'Num samples of x and y do not match'

    def test_base_estimator_calls(self):
        assert issubclass(LinearRegressor, BaseEstimator)
