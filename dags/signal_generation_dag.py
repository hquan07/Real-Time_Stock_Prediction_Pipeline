from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'signal_generation_dag',
    default_args=default_args,
    description='Generate daily trading signals based on model inference and RSI',
    schedule_interval='0 17 * * *',  # Run at 5 PM daily (after market close)
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['trading', 'signals', 'ml'],
) as dag:

    # Define environment variables for the container
    env_vars = {
        **os.environ,
        'POSTGRES_HOST': os.environ.get('POSTGRES_HOST', 'postgres'),
        'POSTGRES_PORT': os.environ.get('POSTGRES_PORT', '5432'),
        'POSTGRES_DB': os.environ.get('POSTGRES_DB', 'stockdb'),
        'POSTGRES_USER': os.environ.get('POSTGRES_USER', 'postgres'),
        'POSTGRES_PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'postgres')
    }

    generate_signals = BashOperator(
        task_id='generate_signals',
        bash_command='python /opt/airflow/scripts/signal_generator.py',
        env=env_vars,
    )

    generate_signals
