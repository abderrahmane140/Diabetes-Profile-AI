import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor


def create_candidates():
    """Create classification pipelines with and without oversampling."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=1,
        ),
        "SVC": SVC(
            kernel="rbf",
            C=1.0,
        ),
    }

    candidates = {}

    for model_name, model in models.items():
        for use_oversampling in [False, True]:

            sampling_name = (
                "oversampling"
                if use_oversampling
                else "no sampling"
            )

            sampler = (
                RandomOverSampler(random_state=42)
                if use_oversampling
                else "passthrough"
            )

            candidate_name = (
                f"{model_name} | {sampling_name}"
            )

            # Create fresh preprocessing steps for each candidate.
            # Unpack them to avoid nesting a Pipeline inside
            # the imbalanced-learn Pipeline.
            candidates[candidate_name] = Pipeline([
                *create_preprocessor().steps,
                ("sampler", sampler),
                ("classifier", clone(model)),
            ])

    return candidates


def compare_classifiers(candidates, X, y, folds):
    """Compare candidates using the same cross-validation folds."""

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            average="macro",
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            average="macro",
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
    }

    results = []

    for name, pipeline in candidates.items():
        print("Evaluating:", name)

        scores = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )

        # "test" in these score keys means the validation fold,
        # not the reserved final test dataset.
        results.append({
            "Model": name,
            "Train_F1": scores["train_f1"].mean(),
            "CV_F1": scores["test_f1"].mean(),
            "CV_F1_std": scores["test_f1"].std(),
            "CV_Precision": scores["test_precision"].mean(),
            "CV_Recall": scores["test_recall"].mean(),
            "CV_Accuracy": scores["test_accuracy"].mean(),
        })

    return (
        pd.DataFrame(results)
        .sort_values("CV_F1", ascending=False)
        .reset_index(drop=True)
    )