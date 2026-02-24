# 📈 Real-Time Stock Prediction Pipeline

A comprehensive real-time stock prediction system using Apache Kafka, Spark, Machine Learning, and Deep Learning.

## 🏗️ Architecture

```mermaid
flowchart LR
    A[YFinance API] --> B[Data Ingestion]
    B --> C[Apache Kafka]
    C --> D[Spark Streaming]
    D --> E[PostgreSQL]
    E --> F[ML/DL Models]
    F --> G[Predictions]
    G --> H[Dashboard]
    I[Apache Airflow] -.-> B
    I -.-> F
```

## 🚀 Features

- **Real-time Data Ingestion**: Fetch stock data from YFinance API
- **Stream Processing**: Apache Kafka + Spark for real-time data processing
- **Machine Learning**: RandomForest, XGBoost models for price prediction
- **Deep Learning**: LSTM neural networks for time series forecasting
- **Interactive Dashboard**: Real-time visualization with Plotly Dash
- **Orchestration**: Apache Airflow for workflow management
- **Containerized**: Docker Compose for easy deployment

## 📁 Project Structure

```
Real-Time_Stock_Prediction_Pipeline/
├── airflow/                    # Airflow DAGs and scripts
│   ├── dags/                   # DAG definitions
│   └── scripts/                # Callable scripts
├── docker/                     # Docker configurations
│   ├── airflow/                # Airflow Dockerfile
│   ├── app/                    # App Dockerfile
│   └── spark/                  # Spark Dockerfile
├── notebooks/                  # Jupyter notebooks for EDA
├── src/                        # Main source code
│   ├── config/                 # Configuration files
│   ├── dashboard/              # Dash web application
│   ├── data_ingestion/         # Data fetching & Kafka producer
│   ├── database/               # PostgreSQL connection & writes
│   ├── deep_learning/          # LSTM models
│   ├── machine_learning/       # ML training & inference
│   ├── streaming/              # Spark streaming jobs
│   └── utils/                  # Helper utilities
├── tests/                      # Unit & integration tests
├── docker-compose.yml          # Docker orchestration
└── requirements.txt            # Python dependencies
```

## 🛠️ Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- PostgreSQL 15

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Real-Time_Stock_Prediction_Pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=Huyquan1607
POSTGRES_DB=stockdb
POSTGRES_PORT=5432

# Kafka
KAFKA_BROKER_ID=1
KAFKA_BROKER_PORT=9092
KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
KAFKA_ADVERTISED_HOST_NAME=kafka

# Spark
SPARK_MASTER_HOST=spark-master

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://postgres:Huyquan1607@postgres:5432/airflow

# Jupyter
JUPYTER_TOKEN=your_token
JUPYTER_PORT=8888
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Initialize Database

```bash
# Run database initialization
python -c "from src.database.db_connection import init_database; init_database()"
```

### 5. Run Components

```bash
# Run data ingestion
python src/data_ingestion/main.py

# Run dashboard
python src/dashboard/app.py

# Train models
python airflow/scripts/run_train.py --model random_forest

# Run inference
python airflow/scripts/run_inference.py
```

## 📊 Dashboard

Access the dashboard at: `http://localhost:8050`

Features:
- Real-time stock price charts (Candlestick/Line)
- ML prediction visualization
- Volume analysis
- Model performance metrics

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Services & Ports

| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 5432 | localhost:5432 |
| Kafka | 9092 | localhost:9092 |
| Zookeeper | 2181 | localhost:2181 |
| Spark Master | 7077, 8080 | http://localhost:8080 |
| Spark Worker | 8082 | http://localhost:8082 |
| Airflow | 8081 | http://localhost:8081 |
| Jupyter | 8888 | http://localhost:8888 |
| Dashboard | 8050 | http://localhost:8050 |

## 📦 Key Dependencies

- **Data**: pandas, numpy, yfinance
- **Streaming**: kafka-python, pyspark
- **ML**: scikit-learn, xgboost
- **DL**: pytorch
- **Database**: sqlalchemy, psycopg2
- **Visualization**: dash, plotly
- **Orchestration**: apache-airflow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.
