from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from src.preprocessing import create_preprocessor


def create_logistic_search(folds):
    pipeline = Pipeline([
        # Flatten the preprocessing steps to avoid the previous error.
        *create_preprocessor().steps,

        ("sampler", "passthrough"),

        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
        )),
    ])

    parameter_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "sampler": [
            "passthrough",
            RandomOverSampler(random_state=42),
        ],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        cv=folds,
        refit=True,
        return_train_score=True,
        n_jobs=1,
        verbose=1,
        error_score="raise",
    )