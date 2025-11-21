import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
)


# ========== CONFIG ==========
DATA_PATH = "data/census.csv"

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

LABEL = "salary"


def main():
    # ===== 1. LOAD DATA =====
    print(f"Loading data from: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    print("Loaded data:", data.shape)

    # ===== 2. TRAIN/TEST SPLIT =====
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    print("Train:", train.shape, " | Test:", test.shape)

    # ===== 3. PROCESS TRAINING DATA =====
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
    )

    # ===== 4. PROCESS TEST DATA =====
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # ===== 5. TRAIN THE MODEL =====
    model = train_model(X_train, y_train)

    # ===== 6. EVALUATE TEST PERFORMANCE =====
    preds = inference(model, X_test)
    p, r, fb = compute_model_metrics(y_test, preds)

    print(f"Test Precision: {p:.4f}")
    print(f"Test Recall:    {r:.4f}")
    print(f"Test F1:        {fb:.4f}")

    # ===== 7. SAVE MODEL ARTIFACTS =====
    save_model(model, encoder, lb)
    print("Model, encoder, and label binarizer saved to /model directory")

    # ===== 8. SLICE METRIC REPORT =====
    print("Computing slice metrics...")

    with open("slice_output.txt", "w") as f:
        for col in CATEGORICAL_FEATURES:
            values = sorted(test[col].unique())

            for val in values:
                slice_df = test[test[col] == val]
                if slice_df.empty:
                    continue

                X_slice, y_slice, _, _ = process_data(
                    slice_df,
                    categorical_features=CATEGORICAL_FEATURES,
                    label=LABEL,
                    training=False,
                    encoder=encoder,
                    lb=lb,
                )

                preds_slice = inference(model, X_slice)
                ps, rs, fbs = compute_model_metrics(y_slice, preds_slice)

                f.write(
                    f"{col} == {val}: "
                    f"precision={ps:.4f}, recall={rs:.4f}, fbeta={fbs:.4f}\n"
                )

    print("Slice metrics written to slice_output.txt")


if __name__ == "__main__":
    main()
