FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
    MLFLOW_DISABLE_TELEMETRY=true MLFLOW_ENABLE_OTLP_EXPORTER=false MLFLOW_DISABLE_AGENT_HINT=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir /state && chown app:app /state
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-compose.txt
ENV GIT_PYTHON_REFRESH=quiet
COPY churn/ churn/
COPY api/ api/
COPY tests/ tests/
USER app
CMD ["python", "-m", "churn.bootstrap", "--state-dir", "/state"]
