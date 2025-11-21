import pickle

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data  # needed for slice performance


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.
    """
    return model.predict(X)


def save_model(
    model,
    encoder,
    lb,
    model_path="model/model.pkl",
    encoder_path="model/encoder.pkl",
    lb_path="model/lb.pkl",
):
    """
    Save the trained model, encoder, and label binarizer to disk.
    """
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    with open(lb_path, "wb") as f:
        pickle.dump(lb, f)


def load_model(
    model_path="model/model.pkl",
    encoder_path="model/encoder.pkl",
    lb_path="model/lb.pkl",
):
    """
    Load the trained model, encoder, and label binarizer from disk.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    with open(lb_path, "rb") as f:
        lb = pickle.load(f)

    return model, encoder, lb


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Computes the model metrics on a slice of the data specified by a column name
    and slice value.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features : list
        List containing the names of the categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model : sklearn estimator
        Trained model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # Filter the slice
    slice_df = data[data[column_name] == slice_value]
    if slice_df.empty:
        return 0.0, 0.0, 0.0

    # Process the slice using the *existing* encoder and label binarizer
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Get predictions for the slice
    preds = inference(model, X_slice)

    # Compute metrics on this slice
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
