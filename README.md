# Diabetes Profile AI

A machine learning project that groups patients by similar clinical
measurements, predicts their cluster, and displays an associated risk category.

## Project objectives

- Explore and clean clinical data.
- Identify patient profiles using K-Means.
- Interpret clusters using the project brief's risk rules.
- Compare at least three classification models.
- Track experiments and register models with MLflow.
- Serve predictions through FastAPI and Streamlit.
- Automate training using Airflow.
- Run the application with Docker Compose.

## Features

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Confirm column names and measurement units against the supplied dataset.

## Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- imbalanced-learn
- MLflow
- FastAPI
- Streamlit
- Apache Airflow
- PostgreSQL
- Docker Compose

## Project organization

| Folder | Purpose |
| --- | --- |
| data/raw/ | Original dataset |
| data/processed/ | Generated datasets and saved split information |
| notebooks/ | Exploration and experiment notebooks |
| src/ | Reusable machine learning functions |
| api/ | Prediction API and input validation |
| app/ | Streamlit interface |
| dags/ | Airflow training workflow |
| docker/ | Service Dockerfiles |
| models/ | Local model exports |
| reports/ | Figures and evaluation results |
| tests/ | Preprocessing and prediction checks |

## Seven-day roadmap

### Day 1 — Setup and exploratory analysis

- [ ] Create the virtual environment and initialize Git.
- [ ] Select compatible Python and dependency versions.
- [ ] Put the dataset in data/raw/diabetes.csv.
- [ ] Load the dataset using pandas.
- [ ] Inspect dimensions, columns, types, and sample rows.
- [ ] Identify duplicates, missing values, and suspicious zeros.
- [ ] Check the dataset documentation for missing-value conventions.
- [ ] Remove confirmed duplicates and reserve the test set.
- [ ] Explore training data with histograms and boxplots.

Deliverable:
01_eda.ipynb and a reproducible train/test split.

### Day 2 — Preprocessing

- [ ] Implement data validation.
- [ ] Convert documented missing-value markers to missing values.
- [ ] Handle missing numerical values using an appropriate strategy.
- [ ] Investigate outliers using IQR and boxplots.
- [ ] Document whether unusual observations are retained or treated.
- [ ] Explore correlations and pairplots on training data.
- [ ] Select useful features and document the choice.
- [ ] Build imputation and scaling into a reusable pipeline.

Fit preprocessing only on training data.
Apply the fitted transformations to validation and test data.

Do not select features simply because their raw variance is larger:
different measurement units affect variance.

Deliverable:
02_preprocessing.ipynb and src/preprocessing.py.

### Day 3 — Clustering and risk interpretation

- [ ] Try several values of k.
- [ ] Plot inertia and silhouette scores.
- [ ] Choose and justify the number of clusters.
- [ ] Train K-Means on preprocessed training data.
- [ ] Add Cluster labels to training observations.
- [ ] Assign test labels using the fitted K-Means model.
- [ ] Calculate cluster sizes and feature means in original units.
- [ ] Apply the risk interpretation rule from the brief.
- [ ] Save the fitted clustering pipeline and risk mapping.

The brief identifies a higher-risk cluster when its means satisfy:
Glucose >126 AND BMI >30 AND DiabetesPedigreeFunction >0.5.

Cluster numbers have no fixed meaning.
Inspect the profiles before mapping clusters to risk categories.
If no cluster meets the rule, document that result.

Deliverable:
03_clustering.ipynb, cluster summaries, and clustering artifacts.

### Day 4 — Classification models

- [ ] Define X using the selected clinical features.
- [ ] Define y using the generated Cluster labels.
- [ ] Keep Cluster and risk_category out of X.
- [ ] Train Logistic Regression.
- [ ] Train Random Forest Classifier.
- [ ] Train Support Vector Classifier.
- [ ] Use the same splits for all models.
- [ ] Check class imbalance.
- [ ] Compare resampling with no resampling when appropriate.
- [ ] Evaluate models using training cross-validation.

Use an imbalanced-learn pipeline when adding a sampler:

Preprocessing → Optional sampler → Classifier

Apply resampling only within training folds.
Do not resample validation or test data.

Deliverable:
Baseline model comparison and reusable training code.

### Day 5 — Tuning and MLflow

- [ ] Tune promising models with GridSearchCV or RandomizedSearchCV.
- [ ] Select the model using a declared cross-validation metric.
- [ ] Evaluate the selected model on the reserved test set.
- [ ] Record accuracy, precision, recall, and F1-score.
- [ ] Save a confusion matrix.
- [ ] Log clustering parameters and metrics in MLflow.
- [ ] Log classifier parameters, metrics, and artifacts.
- [ ] Record feature order, random seed, dataset checksum, and versions.
- [ ] Log the model input/output signature.
- [ ] Save the complete fitted pipeline as a joblib file.
- [ ] Register the pipeline and its matching risk mapping.
- [ ] Mark the validated version as approved for inference.

Use macro-averaged metrics when evaluating multiple clusters.

MLflow note:
The brief requests the Production stage. Model stages are deprecated;
a champion alias is the current alternative. If the assessment requires
the literal Production stage, select a compatible version and document it.

Reference:
https://www.mlflow.org/docs/latest/ml/model-registry/workflow/

Deliverable:
Tracked experiments and a registered model version.

### Day 6 — FastAPI and Streamlit

- [ ] Create a /health endpoint.
- [ ] Create a POST /predict endpoint.
- [ ] Validate patient inputs.
- [ ] Load the approved pipeline from the MLflow Model Registry.
- [ ] Load the risk mapping associated with that exact model version.
- [ ] Return the cluster, risk category, and model version.
- [ ] Build a Streamlit patient form.
- [ ] Send form data to FastAPI.
- [ ] Display the result with a clear visual indicator.
- [ ] Handle invalid inputs and unavailable models.
- [ ] Check that API and local pipeline predictions agree.

Deliverable:
Working patient form and prediction API.

### Day 7 — Airflow, Docker, and demonstration

- [ ] Create an Airflow DAG using reusable functions from src/.
- [ ] Implement data preparation, clustering, and classifier training.
- [ ] Add evaluation and model registration tasks.
- [ ] Promote candidates only when validation checks pass.
- [ ] Save a new risk mapping whenever clustering is retrained.
- [ ] Create Dockerfiles for the application services.
- [ ] Configure Docker Compose and persistent volumes.
- [ ] Use separate metadata databases for MLflow and Airflow.
- [ ] Run the complete training workflow.
- [ ] Refresh the API to load the approved model version.
- [ ] Test the complete application after restart.
- [ ] Document tested startup commands and prepare the demonstration.

Deliverable:
Docker environment, successful Airflow run, and final demonstration.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After populating requirements.txt with compatible dependencies:

```bash
pip install -r requirements.txt
```

Create local environment settings:

```bash
cp .env.example .env
```

Place the dataset here:

```text
data/raw/diabetes.csv
```

Implement and test the services before adding their final startup commands.

## Evaluation rules

- Reserve test data before fitting preprocessing or K-Means.
- Use training cross-validation for classifier selection and tuning.
- Keep resampling inside training folds.
- Preserve feature order and preprocessing during inference.
- Version the cluster-to-risk mapping with the model.
- Keep the final test set out of tuning decisions.

The classifiers learn K-Means cluster assignments.
High classification scores show agreement with those assignments.
They do not establish diagnostic accuracy for diabetes.

## Final deliverables

- EDA and clustering notebooks
- Reusable training code
- Cluster interpretation and risk mapping
- Comparison of at least three classifiers
- Saved inference pipeline
- MLflow experiments and registered model
- FastAPI prediction endpoint
- Streamlit interface
- Airflow training DAG
- Dockerfiles and docker-compose.yml
- README with tested setup instructions