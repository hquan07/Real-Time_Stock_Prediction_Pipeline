#!/bin/bash

set -eu

echo "Starting FastAPI dashboard service..."
cd /app/services/api
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
