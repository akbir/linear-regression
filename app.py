import traceback

import joblib
import numpy as np
from flask import Flask, jsonify, request
import pathlib

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    if model:
        try:
            json = request.json
            data = json['data']
            data = np.asarray(data)
            predictions = model.predict(data)
            return jsonify({'prediction': [np.array2string(y) for y in predictions]})
        except:
            return jsonify({'trace': traceback.format_exc()})

    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    print(pathlib.Path.cwd())
    model = joblib.load('models/model.pkl')
    app.run(debug=True)

