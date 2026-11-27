FROM python:3.12-slim

WORKDIR /app
COPY constraints-python.txt /app/constraints-python.txt
COPY requirements-market-price.txt /app/requirements-market-price.txt
RUN pip install --no-cache-dir -r /app/requirements-market-price.txt && \
    groupadd --system app && useradd --system --gid app --home-dir /nonexistent app
COPY packages/pipeline/src /app/packages/pipeline/src
ENV PYTHONPATH=/app/packages/pipeline/src \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER app
CMD ["python", "-m", "stock_pipeline.ingestion.market_price.worker"]
