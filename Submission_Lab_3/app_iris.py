"""
@author: Carmel Prosper SAGBO
@andrew ID: (csagbo)
"""
import json
from flask import Flask
from flask import request
from flask import Response
from base_iris_lab1 import *

app = Flask(__name__)

employee_names = []
employee_nicknames = []


# the minimal Flask application
@app.route('/')
def index():
    return Response('<h1>Hello, World!</h1>', status=201, content_type="text/html")


# Uploading a new train dataset
@app.route('/iris/datasets', methods=['POST'])
def add_iris_dataset():
    # # Receive and decode CSV data
    data = request.form['train']
    df_index = load_dataset(data)
    print("New Dataset ID comming ", df_index)
    return Response(json.dumps(df_index), status=201, content_type='application/json')


@app.route('/iris/model', methods=['POST'])
def add_iris_model():
    dataset_ID = int(request.form['dataset'])
    try:
        model_ID = new_model(dataset_ID)
    except IndexError:
        return Response(json.dumps("Dataset index error! Please create new dataset !"), status=400,
                        content_type='application/json')

    print("New created model ID comming ", model_ID)
    return Response(json.dumps(model_ID), status=201, content_type='application/json')


@app.route('/iris/model/<n>', methods=['PUT'])
def update_iris_model(n):
    dataset_ID = int(request.args.get('dataset'))
    model_ID = int(n)
    try:
        hist = train(model_ID, dataset_ID)
    except IndexError:
        return Response(json.dumps("Model or dataset index error."), status=400, content_type='application/json')

    return Response(json.dumps(hist), status=201, content_type='application/json')


@app.route('/iris/score/<n>', methods=['GET'])
def get_score_model(n):
    # Extract query parameters for model features
    params = request.args.get('fields')
    fields = []
    for i, value in enumerate(params.split(',')):
        try:
            if i == 1:
                fields.append(int(value))
            else:
                fields.append(float(value))
        except ValueError as e:
            return Response(json.dumps(f"{e}"), status=400, content_type='application/json')

    if len(fields) != 20:
        return Response(json.dumps("List of field mismatch"), status=400, content_type='application/json')

    try:
        sc = score(int(n), fields[0:20])
    except IndexError:
        return Response(json.dumps("No model available at this index."), status=400, content_type='application/json')

    return Response(json.dumps(sc), status=200, content_type='application/json')


@app.route('/iris/model/<n>/test', methods=['GET'])
def test_model(n: int):
    print("chjdchjsc======> ", n, int(request.args.get('dataset')))
    dataset_ID = int(request.args.get('dataset'))
    model_ID = int(n)
    batch_test = test_valid(model_ID, dataset_ID)
    print("cksdbjkbsdkbdskc Red ====> ", batch_test)
    for key, value in batch_test.items():
        if isinstance(value, np.ndarray):
            batch_test[key] = value.tolist()
    return Response(json.dumps(batch_test), status=200, content_type='application/json')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
