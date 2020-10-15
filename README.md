# Linear Regressor

![CoverageBade](images/coverage.svg)

Linear Regression class to extend `sklearn.BaseEstimator` with Ordinary Least Squares.

# Setup

Project uses `Python3.8.0`, to install requirments use:

```pip install -r requirements.txt```

# Usage
Given a file `data.csv` with the format `col_num, x_1, x_2, y`, you can output the following predictions to `predictions.csv`

```python
import numpy as np

from src.linear_regression import LinearRegressor

data = np.genfromtxt('data.csv',
                         delimiter=",",
                         skip_header=1,
                         usecols=(1,2,3))

train, test = data[:25], data[25:]

train_x, train_y = train[:, :2], train[:, 2].reshape(25, 1)
test_x, test_y = test[:, :2], test[:, 2].reshape(25, 1)

model = LinearRegressor()
model.fit(train_x, train_y)
predictions = model.predict(test_x)
np.savetxt('predictions.csv', predictions, delimiter=",")
```

# Development

We use `pytest` for running unit and integration tests:

```bash
pip install -r dev-requirements.tv

# Run unittests
python -m pytest

# Run integration tests
python -m pytest -m integration

# Coverage Report
coverage run --source src -m pytest
coverage report
coverage-badge -o images/coverage.svg
```