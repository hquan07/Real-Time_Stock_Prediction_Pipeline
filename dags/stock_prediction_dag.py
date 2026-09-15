"""
Stock Prediction Pipeline DAG
Fetches stock data and processes it for prediction
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
with DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='Real-time stock data pipeline for prediction',
    schedule_interval='0 9 * * 1-5',  # 9 AM weekdays
    catchup=False,
    tags=['stock', 'prediction', 'pipeline'],
) as dag:

    start = DummyOperator(task_id='start')

    ingest_task = BashOperator(
        task_id='run_data_ingestion',
        bash_command='python /opt/airflow/src/data_ingestion/main.py',
    )

    train_task = BashOperator(
        task_id='train_rf_model',
        bash_command='python /opt/airflow/scripts/run_train.py --model random_forest --days 90',
    )

    end = DummyOperator(task_id='end')

    # Task dependencies
    start >> ingest_task >> train_task >> end