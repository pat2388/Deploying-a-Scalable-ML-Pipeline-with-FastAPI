import pytest
import pandas as pd
from ml.data import process_data
from sklearn.model_selection import train_test_split
from ml.model import train_model
from sklearn.ensemble import RandomForestClassifier
import os
from train_model import cat_features

@pytest.fixture
def data_sample():
    project_path = "./"
    data_path = os.path.join(project_path, "data", "test-data.csv")
    print(data_path)
    data = pd.read_csv(data_path)

    return data

def test_train_model(data_sample):
    """
    Testing the train_model function that it utilizes the Random Forest Classifier
    """
    train, _ = train_test_split(data_sample, test_size=0.2, random_state=50)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_column_match(data_sample):
    """
    Test to check that the provided data has the correct columns matching the feature set
    """
    for feature in cat_features:
        assert feature in data_sample.columns


# TODO: implement the third test. Change the function name and input as needed
def test_not_empty(data_sample):
    """
    Validation that the data set is not empty
    """
    assert not data_sample.empty
    assert data_sample.shape[0] > 0

