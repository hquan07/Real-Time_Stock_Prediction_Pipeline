FROM ubuntu:latest
LABEL authors="hquan07"

ENTRYPOINT ["top", "-b"]

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
COPY ../../requirements.txt /app/requirements.txt

# ------------------------------------------
# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------
# Copy source code
COPY ../../src /app/src

# ------------------------------------------
# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ------------------------------------------
# Set Python path
ENV PYTHONPATH="/app/src"

# ------------------------------------------
# Expose port (useful for dashboard/API)
EXPOSE 8000

# ------------------------------------------
# Default command
CMD ["./start.sh"]