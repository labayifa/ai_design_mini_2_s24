"""
@author: Carmel Prosper SAGBO
@andrew ID: (csagbo)
"""

import logging
import random
import pandas as pd
import requests

API_URL = "http://localhost:6000"


# Uploading a new train dataset
def upload_iris_dataset(data: str):
    """
    Create or upload a new dataset
    :param data:
    :return:
    """
    url = f"{API_URL}/iris/datasets"
    payload = {'train': data}
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload)

    logging.info(response)
    return response


def create_iris_model(dataset_id: int):
    """
    Create new model and train it using the provided Dataset ID
    :param dataset_id: Dataset ID
    :return:
    """
    url = f"{API_URL}/iris/model"
    payload = {'dataset': dataset_id}
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload)
    logging.info(response)
    return response


def retrain_iris_model(model_id: int, dataset_id: int):
    """
    Update the existing model at :param model_id: by training with the dataset at index :param dataset_id:
    :param model_id: Model ID
    :param dataset_id: Dataset ID
    :return:
    """
    url = f"{API_URL}/iris/model/{model_id}"
    headers = {}
    response = requests.put(url, headers=headers, params={'dataset': dataset_id})
    logging.info(response)
    return response


def get_score_model(model_id: int, fields: str):
    """
    Classify class for a given entry
    :param model_id:
    :param fields:
    :return:
    """
    params = {"fields": fields}
    url = f"{API_URL}/iris/score/{model_id}"
    response = requests.get(url, params=params)
    logging.info(response)
    return response


def test_model_data(model_id: int, dataset_id: int):
    url = f"{API_URL}/iris/model/{model_id}/test"
    params = {"dataset": dataset_id}
    response = requests.get(url, params=params)
    logging.info(response)
    return response


if __name__ == "__main__":
    # Upload a new data set
    # Read the dataset
    print("Uploading the dataset.")
    # Read the CSV file into a DataFrame without considering column names
    df = pd.read_csv("iris_extended_encoded.csv")

    # # Convert the entire DataFrame to a string
    df_as_string = '\n'.join(df.apply(lambda row: ','.join(map(str, row)), axis=1))

    upload_response = upload_iris_dataset(f"{df_as_string}")

    print("Dataset upload response: ", upload_response.text)
    #
    # Create a new Model
    print("Creating a new model with the uploaded dataset.")
    model_response = create_iris_model(0)
    print("Model created response: ", model_response.text)

    # Retrain the model with the same dataset
    print("Retraining the model with the uploaded dataset.")
    retrain_response = retrain_iris_model(0, 0)
    print("Retrained model response: ", retrain_response.text)

    # Get the score for a particular entry
    print("Scoring a selected single row.")
    # Define a function to remove the first field from a row
    remove_first_field = lambda row: ','.join(map(str, row[1:]))

    # Apply the function to each row to remove the first field
    df_as_string = df.apply(remove_first_field, axis=1)

    # Choose a random row from the DataFrame to score
    random_row = df_as_string[0]
    score_response = get_score_model(0, f"{random_row}")
    print("Score response: ", score_response.text)

    # TODO: Evaluate batch dataset with a given model
    #
    print("Evaluating the batch dataset with a given model.")
    batch_response = test_model_data(0, 0)
    print("Evaluation result response: ", batch_response.text)

