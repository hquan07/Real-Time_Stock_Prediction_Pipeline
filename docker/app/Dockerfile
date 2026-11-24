# ------------------------------------------
# Base Image
FROM python:3.10-slim

# ------------------------------------------
# Set Work Directory
WORKDIR /app

# ------------------------------------------
# Copy requirements
COPY services/api/requirements.txt /app/requirements.txt
COPY requirements-base.txt /app/requirements-base.txt

# ------------------------------------------
# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------
# Copy source code
COPY src /app/src
COPY services/api /app/services/api

RUN groupadd --system app && \
    useradd --system --gid app --home-dir /nonexistent app && \
    chown -R app:app /app

# ------------------------------------------
# Set Python path
ENV PYTHONPATH="/app/src:/app/services/api"

# ------------------------------------------
# Expose port
EXPOSE 8000

# ------------------------------------------
# Default command
WORKDIR /app/services/api
USER app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
