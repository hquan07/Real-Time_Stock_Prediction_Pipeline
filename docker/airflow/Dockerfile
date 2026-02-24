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

# Use the official way to install additional packages
# This is set as environment variable for entrypoint to handle
ENV _PIP_ADDITIONAL_REQUIREMENTS="yfinance>=0.2.0 pandas>=2.0.0 loguru>=0.7.0"

# DAGs, plugins, scripts are mounted via docker-compose volumes
# Default entrypoint is inherited from base image