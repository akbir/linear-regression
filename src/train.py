import numpy as np
import joblib

from src.linear_regression import LinearRegressor


def run():
    data = np.genfromtxt('data.csv',
                         delimiter=",",
                         skip_header=1,
                         usecols=(1, 2, 3))

    train, test = data[:25], data[25:]

    train_x, train_y = train[:, :2], train[:, 2].reshape(25, 1)
    test_x, test_y = test[:, :2], test[:, 2].reshape(25, 1)

    model = LinearRegressor()
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    np.savetxt('predictions.csv', predictions, delimiter=",")
    joblib.dump(model, 'models/model.pkl')


if __name__ == '__main__':
    run()
