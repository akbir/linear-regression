import numpy as np

from src.linear_regressor import LinearRegressor


def test_1D_regression():
    num_samples = 5
    feature_size = 1
    x = np.random.rand(num_samples, feature_size)
    y = 5 * x
    model = LinearRegressor(y_intercept=False)
    model.fit(x,y)

    y_bar = model.predict(x)
    assert (y_bar == y).all()

def test_base_estimator_func():
    model = LinearRegressor(y_intercept=False)
    params = model.get_params()

def test_regression_fit():
    pass

def test_regression_predict():
    pass

def test_broken_regression():
    pass