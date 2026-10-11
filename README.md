# Diabetes Profile AI

A beginner-friendly machine-learning project that groups patients with similar clinical measurements, trains a classifier to reproduce those groups, and exposes predictions through a web application.

**Educational prototype:** the classifier predicts K-Means cluster labels. Its scores do not measure diabetes diagnostic accuracy. Risk categories are cluster-level rules from the project brief, not clinically validated diagnoses or individual disease probabilities.

## Features

- Exploratory analysis and preprocessing of seven numerical features.
- K-Means clustering and interpretation of patient profiles.
- Comparison of Logistic Regression, Random Forest, and SVC, with and without oversampling.
- Logistic Regression tuning using training cross-validation.
- MLflow experiment tracking and model registration.
- FastAPI prediction endpoint and Streamlit patient form.
- Manual Airflow DAG for retraining.
- Docker Compose for running all four services.

## Architecture

During inference, Streamlit sends patient measurements to FastAPI. FastAPI loads the model identified by the MLflow `champion` alias at startup, then reuses it for predictions.

During retraining, Airflow runs `python -m src.retrain`. The script fits new models, logs validation results, and registers a new version in MLflow with `review_status=pending`. It does not move the `champion` alias.

| Component | Responsibility |
|---|---|
| pandas / NumPy | Data preparation and analysis |
| scikit-learn | Imputation, scaling, clustering, classification, evaluation |
| imbalanced-learn | Random oversampling and classification pipelines |
| MLflow | Experiments, model artifacts, registry, aliases |
| FastAPI | Input validation and predictions |
| Streamlit | Patient form and result display |
| Airflow | Retraining execution and logs |
| Docker Compose | Local service orchestration |

## Project structure

| Path | Contents |
|---|---|
| `notebooks/01_eda.ipynb` | Data exploration |
| `notebooks/02_preprocessing.ipynb` | Preprocessing |
| `notebooks/03_clustering.ipynb` | Cluster selection, fitting, and interpretation |
| `notebooks/04_classification.ipynb` | Baseline model comparison |
| `notebooks/05_optimization.ipynb` | Hyperparameter tuning and final evaluation |
| `notebooks/06_mlflow_tracking.ipynb` | Tracking and initial model registration |
| `src/preprocessing.py` | Feature definitions and preprocessing functions |
| `src/clustering.py` | Clustering utilities |
| `src/training.py` | Classifier creation and comparison |
| `src/optimization.py` | Logistic Regression grid search |
| `src/retrain.py` | Simplified retraining workflow |
| `api/main.py` | FastAPI application |
| `frontend/app.py` | Streamlit application |
| `dags/diabetes_retraining.py` | One-task Airflow DAG |
| `data/processed/` | Raw splits, scaled data, labels, and split metadata |
| `models/` | Locally saved model bundles |
| `reports/metrics/` | Comparison and evaluation CSV files |
| `reports/figures/` | EDA and evaluation figures |
| `mlflow_data/` | Local MLflow SQLite database and artifacts |
| `Dockerfile` | Shared image for MLflow, FastAPI, and Streamlit |
| `Dockerfile.airflow` | Airflow image with a separate training environment |
| `docker-compose.yml` | Complete application stack |
| `compose.airflow.yml` | Earlier standalone Airflow setup; not needed for the full stack |
| `requirements-app.txt` | Pinned application dependencies |
| `requirements-training.txt` | Pinned training dependencies |

Notebook names describe the project workflow; keep them consistent with your repository if you rename files.

## Dataset and preprocessing

The original dataset used in development contains 768 observations. The initial split contains 614 training rows and 154 test rows. The model uses these seven features:

1. `Glucose`
2. `BloodPressure`
3. `SkinThickness`
4. `Insulin`
5. `BMI`
6. `DiabetesPedigreeFunction`
7. `Age`

`Pregnancies` is not used by the current model. Zero pregnancies is a valid value. The sequential index column is not a predictor.

Preprocessing converts zero values to missing values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. Missing values are imputed with training medians, then numerical features are standardized with `StandardScaler`.

The current baseline retains outliers; outlier detection during EDA does not automatically justify removing patients. Preprocessing and oversampling are fitted inside classifier training folds. Oversampling is applied during fitting, not prediction.

The initial K-Means model uses **four clusters**. The classifier target is `Cluster`; neither `Cluster` nor `risk_category` is an input feature.

## Risk mapping

A cluster is labeled high risk when all three cluster means exceed the assignment thresholds:

- Glucose > 126
- BMI > 30
- Diabetes Pedigree Function > 0.5

Otherwise, it receives the assignment's low-risk label. These means are computed in original measurement units, after missing-value handling and before scaling.

Cluster IDs are arbitrary and may change after retraining. Each classifier version therefore includes its own `risk_mapping_` and `feature_list_`. The frontend normalizes category capitalization before displaying the result.

## Reported model results

The selected configuration is **Logistic Regression, C=1.0, with RandomOverSampler**. Grid search evaluated ten configurations over five folds and selected the same configuration as the baseline.

| Metric | Reported value |
|---|---:|
| Mean training macro F1 across CV folds | 0.9810 |
| Mean validation macro F1 across CV folds | 0.9750 |
| CV macro F1 standard deviation | 0.0110 |
| Held-out test accuracy | 0.968 |
| Held-out test macro precision | 0.946 |
| Held-out test macro recall | 0.928 |
| Held-out test macro F1 | 0.936 |

The test model correctly predicted 149 of 154 cluster labels. Cluster 2 was the weakest class: test F1 was 0.824, based on only nine patients.

The training F1 above is an average over CV training folds, not the final model's score on the complete training set. The CV labels come from a K-Means model fitted on the initial training set, so CV evaluates classifiers against a fixed clustering model; it is not an independent evaluation of the entire clustering-discovery process.

These are the original reported results. New retraining runs use a different fitting/validation partition and generate new cluster targets, so their scores are not directly interchangeable with this table.

## Run the prepared project with Docker

### Prerequisites

- Docker Engine and Docker Compose v2.
- Project source files and both requirements files.
- Existing `data/processed/train_raw.csv` for retraining.
- Existing `mlflow_data/mlflow.db` and its `artifacts/` directory.
- A registered `diabetes-profile-classifier` version with alias `champion` for API startup.

From the project root:

```bash
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

| Service | Local URL |
|---|---|
| Streamlit | http://localhost:8501 |
| FastAPI documentation | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

The containers communicate using service names: FastAPI and Airflow use `http://mlflow:5000`; Streamlit uses `http://api:8000`. Browser access uses the localhost URLs above.

The current images use Python 3.12; use the pinned requirements when rebuilding. This is a local learning setup with SQLite and Airflow standalone, not a production deployment.

### First setup without trained artifacts

The full stack does not train or create a champion automatically. On a clean checkout:

1. Create a local Python environment and install the project dependencies plus `jupyter` and `ipykernel`. Install `matplotlib` and `seaborn` for EDA and plots if they are not already present.
2. Place the historical dataset in the location expected by the EDA notebook.
3. Run notebooks 01–05 in order to create splits, clusters, and the final classifier bundle.
4. Start only the MLflow service with `docker compose up --build -d mlflow`.
5. Run notebook 06 against `http://127.0.0.1:5000` to log models, verify the registered pipeline, and assign the initial `champion` alias.
6. Start the remaining services with `docker compose up --build -d`.

Keep the original test set out of model selection and routine retraining decisions. This README does not include patient data or trained artifacts.

## Try a prediction

Enter this fictional example in Streamlit or POST it to `/predict`:

```json
{
  "Glucose": 140,
  "BloodPressure": 80,
  "SkinThickness": 25,
  "Insulin": 100,
  "BMI": 32.5,
  "DiabetesPedigreeFunction": 0.6,
  "Age": 45
}
```

The current application returned cluster **3** with category **low risk** for this example. That is an observed output from the current model, not a guaranteed output of future versions.

API input rules:

- All seven keys are required.
- The first five features accept `null` for missing measurements.
- Age must be positive; other numerical values must be nonnegative.
- Extra fields and nonfinite numerical values are rejected.
- The frontend additionally requires at least one nonzero measurement among the first five features.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Report readiness after model loading |
| POST | `/predict` | Return cluster, risk category, and explanatory note |

## Retraining with Airflow

Retrieve the generated Airflow password:

```bash
docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Log in as `admin` using the generated password. The configuration value `admin:admin` means username and role, not username and password.

1. Open the `diabetes_retraining` DAG.
2. Enable it if paused.
3. Trigger a manual run.
4. Inspect the `train_model` task logs.
5. Check MLflow for a `simple_retraining` run and a new registered version.

The DAG has one task, no automatic schedule, and no automatic retries. It runs `python -m src.retrain` in the container's separate training environment.

The simplified script splits the existing training CSV into fitting and validation portions, refits K-Means with k=4, recomputes risk categories, and retrains Logistic Regression with oversampling and C=1.0. It logs validation accuracy, macro F1, cluster profiles, risk mapping, and the classifier pipeline. It does not repeat the full classifier comparison or grid search.

The script currently reads a fixed CSV. Collecting new patient data and refreshing that file are not automated. Repeating training on unchanged data does not inherently improve the model.

## Review and promotion

New retrained versions are tagged `review_status=pending`. Automatic promotion is not implemented.

Before moving the `champion` alias:

1. Review validation results, class coverage, cluster profiles, and the associated risk mapping.
2. Verify that the candidate loads correctly and accepts the API's feature schema.
3. Record the previous champion version so it can be restored.
4. Assign `champion` to the reviewed version in the MLflow registry.
5. Restart the API and verify prediction behavior:

```bash
docker compose restart api
```

The API resolves the alias only at startup. Changing it does not hot-reload the running API. Rollback follows the same process: restore the previous alias target and restart the API.

This project uses the `champion` alias rather than the deprecated MLflow Production stage. A higher cluster-prediction score alone is not sufficient evidence that newly discovered profiles are better.

## Useful commands

```bash
# View all containers
docker compose ps

# View recent API logs
docker compose logs --tail=100 api

# Follow Airflow service logs
docker compose logs -f airflow

# Rebuild the frontend after editing its copied source
docker compose up -d --build --no-deps frontend

# Rebuild the API after editing its copied source
docker compose up -d --build --no-deps api

# Stop containers while preserving saved data
docker compose down

# Start again
docker compose up -d
```

FastAPI and Streamlit source is copied into their image, so changes require rebuilding. Airflow mounts `src/`, `dags/`, and `data/` read-only from the host.

MLflow data lives in the host's `mlflow_data/` directory. Airflow state lives in the Compose named volume `airflow_data`. Avoid `docker compose down -v` when preserving Airflow history.

For a consistent local MLflow backup, wait for training to finish, stop the stack, copy the directory, and start again:

```bash
docker compose down
cp -a mlflow_data "mlflow_data.backup-$(date +%Y%m%d-%H%M%S)"
docker compose up -d
```

## Troubleshooting

| Problem | What to check |
|---|---|
| MLflow connection refused | Ensure MLflow is running; containers must use `http://mlflow:5000`, not their own localhost. |
| API cannot find the model or alias | Verify the registered model name and `champion` alias in the mounted MLflow database. |
| Airflow rejects `admin/admin` | Read the generated password using the command above. |
| Streamlit shows an unrecognized risk category | Normalize the returned label with `.strip().lower()` and compare with `high risk` / `low risk`. |
| A source-code edit has no effect | Rebuild the affected application service. |
| Missing `logistic_tuning.csv` in notebook 06 | Save `tuning_results` from notebook 05 into `reports/metrics/logistic_tuning.csv`. |
| MLflow raises an untrusted-types error | Use the explicit reviewed skops type list in the model-logging code; do not automatically trust arbitrary types. |
| Retraining fails | Inspect the Airflow task logs and confirm the training CSV is present and readable. |

## Seven-day learning roadmap

| Day | Goal |
|---|---|
| 1 | Explore data, check quality, and preserve a test split |
| 2 | Handle missing values, inspect outliers, and build preprocessing |
| 3 | Select k, fit K-Means, and interpret cluster profiles |
| 4 | Compare classifiers, tune Logistic Regression, and evaluate |
| 5 | Track experiments and register the initial model in MLflow |
| 6 | Build FastAPI and Streamlit, then test predictions |
| 7 | Add retraining, Airflow, Docker Compose, and documentation |

## Current verification and remaining work

Confirmed during development:

- All four Compose containers were running; MLflow and FastAPI passed health checks.
- Streamlit successfully displayed the example prediction after label normalization.
- The original classifier bundle was saved and reloaded with matching predictions.

Final checks still to record:

- Successful retraining through the full Compose stack and creation of a pending model version.
- Candidate loading verification, reviewed promotion, and rollback exercise.
- Audit of the original brief's required plots and per-model tracking artifacts.

The simplified retraining script logs fewer artifacts than the initial notebooks: it does not currently persist its refitted K-Means model or repeat all classifier experiments. Extend that script if every retraining run must provide the same full traceability as the initial training workflow.

## Limitations

- The target is a generated cluster label, not a confirmed diabetes outcome.
- The risk rule is a classroom heuristic and does not establish clinical safety or effectiveness.
- The minority cluster has limited evaluation support.
- The current system has no automatic new-data ingestion, drift monitoring, or promotion policy.
- Deployment uses local development services; access control and production infrastructure are outside this implementation.

## Documentation

- [scikit-learn](https://scikit-learn.org/stable/)
- [imbalanced-learn](https://imbalanced-learn.org/stable/)
- [MLflow](https://mlflow.org/docs/latest/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [Docker Compose](https://docs.docker.com/compose/)