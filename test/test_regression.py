import numpy as np
import pytest
from unittest import mock
from src.linear_regression import LinearRegressor


def _create_linear_values(num_samples, feature_size, coefficient, bias):
    x = np.random.rand(num_samples, feature_size)
    y = coefficient * x + bias
    return x, y


@pytest.mark.integration
@pytest.mark.parametrize('num_samples,features,coefficient,bias',
                         [(20, 1, 5, 0),
                          (100, 25, 0.1, 0),
                          (20, 1, 5, 10),
                          (1, 1, 5, 10),
                          ]
                         )
def test_linear_regression(num_samples, features, coefficient, bias):
    x, y = _create_linear_values(num_samples, features, coefficient, bias)
    model = LinearRegressor()
    model.fit(x, y)
    y_bar = model.predict(x)
    np.testing.assert_almost_equal(y_bar, y, decimal=6)

@pytest.mark.parametrize('num_samples,features,coeff,bias',
                         [(20, 10, 3, 0),
                          (100, 25, 5, 0),
                          (20, 1, 7, 10),
                          (3, 1, 5, 0.5),
                          ]
                         )
@mock.patch("src.linear_regression.invert_matrix", autospec=True)
def test_regression_fit(mock_invert, num_samples, features, coeff, bias):
    # mock out our custom matrix inverse
    mock_invert.side_effect = lambda z: np.linalg.inv(z)
    x, y = _create_linear_values(num_samples, features, coeff, bias)
    model = LinearRegressor()
    model.fit(x,y)

    # assert some things here.
    mock_invert.assert_called_once()
    np.testing.assert_almost_equal(model._weights[1:], coeff * np.eye(features, features))
    np.testing.assert_almost_equal(model._weights[0], bias * np.ones((features,)))


@pytest.mark.parametrize('num_samples,features,coeff,bias',
                         [(20, 10, 3, 0),
                          (100, 25, 5, 0),
                          (20, 1, 7, 10),
                          (3, 3, 5, 0.5),
                          ]
                         )
def test_regression_predict(num_samples, features, coeff, bias):
    x, y = _create_linear_values(num_samples, features, coeff, bias)
    model = LinearRegressor()

    # mock fit
    model._fit = True
    model._weights = np.concatenate([
        bias * np.ones((1, features)),
        coeff * np.eye(features, features),
    ], axis=0)

    y_bar = model.predict(x)
    np.testing.assert_almost_equal(y, y_bar)


def test_predict_before_train_error():
    model = LinearRegressor()
    with pytest.raises(ValueError) as e:
        model.predict(np.zeros((10,2)))
    assert str(e.value) == 'Model has not been fit yet, run .fit() first'


def test_base_estimator():
    model = LinearRegressor()
    params = model.get_params()
    assert params == {'weights': None}

    model.set_params(weights=5)
    assert model.get_params() == {'weights': 5}
    assert model._weights == 5