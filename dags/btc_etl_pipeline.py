"""Daily BTC-USD ETL -> feature engineering -> retrain/evaluate -> API reload."""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.environ.get("PYTHONPATH", "/opt/airflow/project"))

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/shared/data")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/airflow/shared/models")
RELOAD_URL = os.environ.get("API_RELOAD_URL", "http://btc-forecast-svc/reload")
RAW = os.path.join(DATA_DIR, "btc_raw.csv")
FEATURES = os.path.join(DATA_DIR, "btc_features.csv")


def fetch_data():
    from src.etl_fetch import main

    main(RAW, period="5y")


def preprocess():
    import pandas as pd

    from src.forecast_lib import add_features

    df = pd.read_csv(RAW, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated()].ffill().dropna()
    add_features(df).dropna().to_csv(FEATURES)
    print(f"[preprocess] wrote {FEATURES}")


def retrain_and_evaluate(**ctx):
    from src.train_forecast import train

    m = train(RAW, MODEL_DIR, mape_threshold=25.0)
    ctx["ti"].xcom_push(key="metrics", value=m)
    if not m["passed"]:
        raise ValueError(f"Model failed threshold: {m}")


def reload_api():
    import requests

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    r = requests.post(RELOAD_URL, headers=headers, timeout=30)
    r.raise_for_status()
    print(f"[reload] {r.json()}")


with DAG(
    dag_id="btc_etl_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["btc", "mlops"],
) as dag:
    t1 = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t2 = PythonOperator(task_id="preprocess_features", python_callable=preprocess)
    t3 = PythonOperator(task_id="retrain_evaluate", python_callable=retrain_and_evaluate)
    t4 = PythonOperator(task_id="reload_api_model", python_callable=reload_api)
    t1 >> t2 >> t3 >> t4
