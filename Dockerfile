# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Build tools occasionally required by scientific wheels (catboost/shapely/polars)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.1.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# The app resolves geo data via $PROJECT_PATH/iep_calculator/data/...
# so the repo is placed at /app/iep_calculator and PROJECT_PATH points to /app.
ENV PROJECT_PATH=/app
# Provide a real Yandex Maps JS / Geocoder key at run time:
#   docker run -e YANDEX_MAPS_API_JS_KEY=<key> ...
ENV YANDEX_MAPS_API_JS_KEY=""

WORKDIR /app/iep_calculator

# Install dependencies first for better layer caching
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the project (model + hedonic index parquet included)
COPY . .

# uvicorn is launched from src/ so `apps.api:app` and the top-level modules resolve
WORKDIR /app/iep_calculator/src

EXPOSE 8000
CMD ["uvicorn", "apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
