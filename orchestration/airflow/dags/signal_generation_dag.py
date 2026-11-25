from datetime import datetime, timedelta

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

    generate_signals = BashOperator(
        task_id='generate_signals',
        bash_command='python /opt/airflow/scripts/signal_generator.py',
    )

    publish_signals = BashOperator(
        task_id='publish_signal_report_to_cloudvault',
        bash_command=(
            'python /opt/airflow/scripts/publish_signals.py '
            '--logical-date "{{ ds }}" '
            '--source-run-id "{{ run_id }}"'
        ),
    )

    generate_signals >> publish_signals
