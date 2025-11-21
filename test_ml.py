import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
)


# Small dummy dataset that looks like census data
def _make_dummy_data():
    data = pd.DataFrame(
        {
            "age": [25, 40, 35],
            "workclass": ["Private", "Self-emp-not-inc", "Private"],
            "fnlgt": [226802, 89814, 336951],
            "education": ["11th", "Bachelors", "HS-grad"],
            "education-num": [7, 13, 9],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
            ],
            "occupation": ["Machine-op-inspct", "Exec-managerial", "Handlers-cleaners"],
            "relationship": ["Own-child", "Husband", "Not-in-family"],
            "race": ["Black", "White", "White"],
            "sex": ["Male", "Male", "Female"],
            "capital-gain": [0, 0, 0],
            "capital-loss": [0, 0, 0],
            "hours-per-week": [40, 60, 40],
            "native-country": ["United-States", "United-States", "United-States"],
            "salary": ["<=50K", ">50K", "<=50K"],
        }
    )

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = "salary"
    return data, cat_features, label


def test_process_data_shapes_match():
    """X and y should have the same number of rows after processing."""
    data, cat_features, label = _make_dummy_data()

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label=label,
        training=True,
    )

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model_and_inference():
    """Model should train and produce predictions of the correct shape."""
    data, cat_features, label = _make_dummy_data()

    X_train, y_train, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label=label,
        training=True,
    )

    model = train_model(X_train, y_train)

    # Run inference on a subset
    X_sample = X_train[:2]
    preds = inference(model, X_sample)

    assert preds.shape[0] == X_sample.shape[0]
    # predictions should be 0/1 values
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_output_types():
    """compute_model_metrics should return floats between 0 and 1."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0
