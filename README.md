# Linear Regressor

![CoverageBade](images/coverage.svg)

Linear Regression class to extend `sklearn.BaseEstimator` with Ordinary Least Squares.

# Setup

Project uses `Python3.8.0`, to install requirments use:

```bash
git clone https://github.com/akbir/linear-regression.git
cd linear-regression
pip install -r requirements.txt
```

# Usage

Linear Regressor can be used in the following ways

1. [Module](#module)
2. [Web App](#web-app)
3. [Docker Deployment](#Docker)


### Module

The module api follows that of Sklearn. For a full example check `examples/train.py`
```python
import numpy as np

from src.linear_regression import LinearRegressor

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegressor()
model.fit(X, y)

model.predict(np.array([[3,5]]))
# [[16.]]
```

### Web App

We can serve a pre-trained model (stored in `models/`) as a Python Micro App. 
To serve a model on port `5000`, run the following:

```bash
python app.py
```
The app currently serves the following endpoints:
 
`/predict` - get predictions for your model:

```bash
 curl -i -X POST -H 'Content-Type: application/json' -d '{"data": [[0,1],[2,3]]}' http://127.0.0.1:5000/predict
```

with expected response

```
{
  "prediction": [
    "[0.30583531]", 
    "[1.77134556]"
  ]
}
```

### Docker
We can also serve our app in a clean docker container!
```
docker build -t lr:latest .
docker run -d -p 5000:5000 lr
```

# Development

We use `pytest` for running unit and integration tests:

```bash
pip install -r dev-requirements.tv

# Run unittests
python -m pytest -m "not integration"

# Run unittests and integrations tests
python -m pytest

# Coverage Report
coverage run --source src -m pytest
coverage report
coverage-badge -o images/coverage.svg
```