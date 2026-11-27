# Spark Dockerfile for Real-Time Stock Prediction Pipeline
# Using official Apache Spark Docker image for fast builds
FROM apache/spark:3.5.3

USER root

# ---------------------------------------------------
# ENV
# ---------------------------------------------------
ENV SPARK_HOME=/opt/spark \
    PATH=/opt/spark/bin:/opt/spark/sbin:$PATH

# ---------------------------------------------------
# Install additional dependencies if needed
# ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-spark.txt /tmp/requirements-spark.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-spark.txt

RUN mkdir -p /home/spark/.ivy2/cache /home/spark/.ivy2/jars /opt/spark/checkpoints \
    && chown -R spark:spark /home/spark/.ivy2 /opt/spark/checkpoints

# ---------------------------------------------------
# Entry
# ---------------------------------------------------
WORKDIR /opt/spark

USER spark

CMD ["bash"]
