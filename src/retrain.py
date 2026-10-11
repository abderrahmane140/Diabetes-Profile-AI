import os
from pathlib import Path
from importlib.metadata import version

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURES, create_preprocessor
from src.training import create_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "diabetes-profile-classifier"


def retrain():
    # 1. Connect to MLflow.
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment("diabetes-profile-ai")

    # 2. Load our existing training data.
    data = pd.read_csv(
        PROJECT_ROOT / "data/processed/train_raw.csv",
        index_col="source_row",
    )

    X = data.loc[:, FEATURES].astype(float)

    # Keep a validation portion. Do not use the original test set.
    X_train, X_validation = train_test_split(
        X,
        test_size=0.2,
        random_state=42,
    )

    # 3. Train K-Means and generate cluster labels.
    clustering = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("kmeans", KMeans(
            n_clusters=4,
            n_init=10,
            random_state=42,
        )),
    ])

    clustering.fit(X_train)

    y_train = clustering.predict(X_train)
    y_validation = clustering.predict(X_validation)

    if len(set(y_train)) != 4:
        raise ValueError("K-Means did not create four distinct clusters.")

    # 4. Calculate cluster means before scaling.
    preprocessor = clustering.named_steps["preprocessor"]

    clean_data = pd.DataFrame(
        preprocessor[:-1].transform(X_train),
        columns=FEATURES,
        index=X_train.index,
    )

    clean_data["Cluster"] = y_train
    cluster_means = clean_data.groupby("Cluster").mean()

    risk_mapping = {}

    for cluster, row in cluster_means.iterrows():
        high_risk = (
            row["Glucose"] > 126
            and row["BMI"] > 30
            and row["DiabetesPedigreeFunction"] > 0.5
        )

        risk_mapping[int(cluster)] = (
            "High risk" if high_risk else "Low risk"
        )

    # 5. Retrain the classifier we already selected.
    model = create_candidates()["Logistic Regression | oversampling"]
    model.fit(X_train, y_train)

    # Metadata required by our FastAPI application.
    model.feature_list_ = list(FEATURES)
    model.risk_mapping_ = risk_mapping

    # 6. Evaluate on validation data.
    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    f1 = f1_score(
        y_validation,
        predictions,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0,
    )

    # 7. Record the results and save the model.
    with mlflow.start_run(run_name="simple_retraining"):
        mlflow.log_params({
            "k": 4,
            "classifier": "Logistic Regression",
            "C": model.named_steps["classifier"].C,
            "sampling": "RandomOverSampler",
            "random_state": 42,
        })

        mlflow.log_metrics({
            "validation_accuracy": accuracy,
            "validation_f1_macro": f1,
        })

        mlflow.log_table(
            cluster_means.reset_index(),
            artifact_file="cluster_profiles.json",
        )

        mlflow.log_dict(
            {str(key): value for key, value in risk_mapping.items()},
            "risk_mapping.json",
        )

        packages = [
            "mlflow", "scikit-learn", "imbalanced-learn",
            "pandas", "numpy", "scipy", "joblib", "skops",
        ]

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="pipeline",
            serialization_format="skops",
            skops_trusted_types=[
                "imblearn.over_sampling._random_over_sampler.RandomOverSampler",
                "imblearn.pipeline.Pipeline",
                "numpy.dtype",
                "src.preprocessing.prepare_features",
            ],
            signature=infer_signature(X_validation, predictions),
            code_paths=[str(PROJECT_ROOT / "src")],
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in packages
            ],
        )

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
            tags={"review_status": "pending"},
        )

    print(f"Validation accuracy: {accuracy:.3f}")
    print(f"Validation macro F1: {f1:.3f}")
    print("Registered version:", registered.version)
    print("Your current champion remains unchanged.")

    return registered.version


if __name__ == "__main__":
    retrain()