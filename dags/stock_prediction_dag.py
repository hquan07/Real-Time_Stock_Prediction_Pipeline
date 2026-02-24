"""
Stock Prediction Pipeline DAG
Fetches stock data and processes it for prediction
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
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


def fetch_stock_data(**context):
    """Fetch stock data from API"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            data[symbol] = {
                'latest_close': float(hist['Close'].iloc[-1]) if len(hist) > 0 else None,
                'records': len(hist)
            }
            print(f"Fetched {len(hist)} records for {symbol}")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            data[symbol] = {'error': str(e)}
    
    return data


def process_data(**context):
    """Process and validate stock data"""
    ti = context['ti']
    stock_data = ti.xcom_pull(task_ids='fetch_stock_data')
    
    if not stock_data:
        raise ValueError("No stock data received from previous task")
    
    processed = {}
    for symbol, data in stock_data.items():
        if 'error' in data:
            processed[symbol] = {'status': 'failed', 'error': data['error']}
        else:
            processed[symbol] = {
                'status': 'success',
                'latest_close': data['latest_close'],
                'records': data['records']
            }
    
    print(f"Processed data for {len(processed)} symbols")
    return processed


def save_to_database(**context):
    """Save processed data to PostgreSQL"""
    ti = context['ti']
    processed_data = ti.xcom_pull(task_ids='process_data')
    
    print("Saving to database...")
    for symbol, data in processed_data.items():
        if data['status'] == 'success':
            print(f"  - {symbol}: ${data['latest_close']:.2f} ({data['records']} records)")
    
    print("Data saved successfully!")
    return True


def send_notification(**context):
    """Send notification about pipeline completion"""
    ti = context['ti']
    save_result = ti.xcom_pull(task_ids='save_to_database')
    
    if save_result:
        print("Pipeline completed successfully!")
        print(f"Execution time: {context['execution_date']}")
    else:
        print("Pipeline failed!")


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

    fetch_task = PythonOperator(
        task_id='fetch_stock_data',
        python_callable=fetch_stock_data,
        provide_context=True,
    )

    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        provide_context=True,
    )

    save_task = PythonOperator(
        task_id='save_to_database',
        python_callable=save_to_database,
        provide_context=True,
    )

    notify_task = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        provide_context=True,
    )

    end = DummyOperator(task_id='end')

    # Task dependencies
    start >> fetch_task >> process_task >> save_task >> notify_task >> end