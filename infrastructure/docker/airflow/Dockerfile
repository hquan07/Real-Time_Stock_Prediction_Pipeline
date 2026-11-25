FROM apache/airflow:2.9.3-python3.10

# Switch to root to install OS dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    libsasl2-dev \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# DAGs, plugins, scripts are mounted via docker-compose volumes
# Default entrypoint is inherited from base image
