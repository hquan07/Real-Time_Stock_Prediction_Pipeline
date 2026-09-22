# ------------------------------------------
# Base Image
FROM python:3.10-slim

# ------------------------------------------
# Set Work Directory
WORKDIR /app

# ------------------------------------------
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------
# Copy requirements
COPY src/dashboard/backend/requirements.txt /app/requirements.txt
COPY requirements-base.txt /app/requirements-base.txt

# ------------------------------------------
# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir psycopg2-binary pandas loguru

# ------------------------------------------
# Copy source code
COPY src /app/src

# ------------------------------------------
# Set Python path
ENV PYTHONPATH="/app/src"

# ------------------------------------------
# Expose port
EXPOSE 8000

# ------------------------------------------
# Default command
WORKDIR /app/src/dashboard/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]