from contextlib import asynccontextmanager

import os
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# MLflow configuration
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

MODEL_URI = "models:/diabetes-profile-classifier@champion"


# Expected patient data
class PatientInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    Glucose: float | None = Field(ge=0)
    BloodPressure: float | None = Field(ge=0)
    SkinThickness: float | None = Field(ge=0)
    Insulin: float | None = Field(ge=0)
    BMI: float | None = Field(ge=0)
    DiabetesPedigreeFunction: float = Field(ge=0)
    Age: float = Field(gt=0)


# Load the model once when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = mlflow.sklearn.load_model(MODEL_URI)

    # Verify that the saved metadata matches the API.
    if set(pipeline.feature_list_) != set(PatientInput.model_fields):
        raise ValueError("Model features do not match the API fields.")

    if not all(
        int(cluster) in pipeline.risk_mapping_
        for cluster in pipeline.classes_
    ):
        raise ValueError("The model is missing a cluster-to-risk mapping.")

    app.state.pipeline = pipeline

    print("Model loaded successfully.")
    yield


app = FastAPI(
    title="Diabetes Profile API",
    description="Predict patient clusters and their associated risk categories.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/predict")
def predict(patient: PatientInput):
    pipeline = app.state.pipeline

    # Use the same column order as training.
    patient_data = pd.DataFrame(
        [patient.model_dump()],
        columns=pipeline.feature_list_,
        dtype=float,
    )

    # The pipeline performs preprocessing and classification.
    cluster = int(pipeline.predict(patient_data)[0])
    risk_category = pipeline.risk_mapping_[cluster]

    return {
        "cluster": cluster,
        "risk_category": risk_category,
        "note": "Cluster-based risk category, not a diabetes diagnosis.",
    }


# python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000