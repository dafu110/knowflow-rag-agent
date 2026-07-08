FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KNOWFLOW_STORE_BACKEND=sqlite \
    KNOWFLOW_STORE=/app/data/knowflow.db

WORKDIR /app

COPY pyproject.toml README.md ./
COPY knowflow ./knowflow
COPY sample_docs ./sample_docs
COPY evals ./evals

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[prod]"

RUN mkdir -p /app/data

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import json, urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3))['ok']"

CMD ["gunicorn", "knowflow.wsgi:application", "--bind", "0.0.0.0:8765", "--workers", "2"]
