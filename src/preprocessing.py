import numpy as np
import pandas as pd


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler



FEATURES = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

def prepare_features(data):
    """ Select features, validate numbers, and replace missing-value zeros. """

    missing_columns = [
        column
        for column in FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select features in a fixed order without modifying the original data.
    features = data.loc[:, FEATURES].copy()

    # Raise an error for unexpected text instead of silently accepting it.
    features = features.apply(
        pd.to_numeric,
        errors="raise"
    ).astype(float)

    if np.isinf(features.to_numpy()).any():
        raise ValueError("Features contain infinite values.")


    if features.lt(0).any().any():
        raise ValueError("Features contain negative measurements.")

    features[ZERO_AS_MISSING] = (
        features[ZERO_AS_MISSING].replace(0, np.nan)
    )

    return features



def create_preprocessor():
    """build an unfitted preprocessing pipeline."""

    return Pipeline([
        (
            "prepare",
            FunctionTransformer(
                prepare_features,
                validate=False
            ),
        ),
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler(),
        )
    ])