from datetime import datetime, timezone
import subprocess

from airflow.sdk import dag, task


@dag(
    dag_id="diabetes_retraining",
    start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["diabetes", "training"],
)

def diabetes_retraining():

    @task(retries=0)
    def train_model():
        subprocess.run(
            [
                "/home/airflow/training-env/bin/python",
                "-m",
                "src.retrain",
            ],
            cwd="/opt/project",
            check=True
        )

    train_model()


diabetes_retraining()






