# ------------------------------------------
# Base Image
FROM python:3.12-slim

RUN apt-get update && \
    apt-get upgrade --yes && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------
# Set Work Directory
WORKDIR /app

# ------------------------------------------
# Copy requirements
COPY services/api/requirements.txt /app/requirements.txt

# ------------------------------------------
# Install Python packages
RUN pip install --no-cache-dir --upgrade pip==26.3 setuptools==84.2.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall --yes pip setuptools

# ------------------------------------------
# Copy source code
COPY src /app/src
COPY packages/pipeline/src /app/packages/pipeline/src
COPY infrastructure/migrations /app/infrastructure/migrations
COPY services/api /app/services/api

RUN groupadd --system app && \
    useradd --system --gid app --home-dir /nonexistent app && \
    chown -R app:app /app

# ------------------------------------------
# Set Python path
ENV PYTHONPATH="/app/packages/pipeline/src:/app/src:/app/services/api" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ------------------------------------------
# Expose port
EXPOSE 8000

# ------------------------------------------
# Default command
WORKDIR /app/services/api
USER app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
