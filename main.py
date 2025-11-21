import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY: input schema
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# ===== Load model artifacts at startup =====
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")

model, encoder, lb = load_model(
    model_path=MODEL_PATH,
    encoder_path=ENCODER_PATH,
    lb_path=LB_PATH,
)


# ===== Create FastAPI app =====
app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income is <=50K or >50K based on census data.",
    version="1.0.0",
)


# ===== Root endpoint =====
@app.get("/")
async def get_root():
    """Simple health check / welcome endpoint."""
    return {"message": "Welcome to the census income prediction API."}


# ===== Prediction endpoint =====
@app.post("/predict")
async def post_inference(data: Data):
    """
    Run model inference on a single record.
    """
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # Field names use underscores, but original data uses hyphens.
    data_clean = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(data_clean)

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

    # Process the incoming data using the existing encoder / lb
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run inference
    preds = inference(model, X)

    # Convert numeric prediction(s) to label(s)
    labels = apply_label(preds)

    # Single row → return single label
    return {"result": labels[0]}
