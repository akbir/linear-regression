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

1. [Module](###Module)
2. [Web App](###WebApp)
3. [Docker Deployment](#Docker)


### Module

Given a file `data.csv` with the format `col_num, x_1, x_2, y`, you can output the following predictions to `predictions.csv`

```python
import numpy as np

from main.linear_regression import LinearRegressor


data = np.genfromtxt('data.csv', delimiter=",",
                         skip_header=1,usecols=(1,2,3))

train, test = data[:25], data[25:]

train_x, train_y = train[:, :2], train[:, 2].reshape(25, 1)
test_x, test_y = test[:, :2], test[:, 2].reshape(25, 1)

model = LinearRegressor()
model.fit(train_x, train_y)
predictions = model.predict(test_x)
np.savetxt('predictions.csv', predictions, delimiter=",")
```

### Web App

Given a port run the following to start a Python Web App exposed on port `8000`:

```bash
python app.py --port 8000 

 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
The app currently serves the following endpoints:
 
`/predict` - get predictions for your model:

```bash
 curl -i -X POST -H 'Content-Type: application/json' -d '{"data": [[0,1],[2,3]]}' http://127.0.0.1:5000/predict
```

with expected response

```

HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 66
Server: Werkzeug/1.0.1 Python/3.8.0

{
  "prediction": [
    "[0.30583531]", 
    "[1.77134556]"
  ]
}

```

### Docker




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