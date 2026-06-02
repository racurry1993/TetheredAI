FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_cloudrun.txt /app/requirements_cloudrun.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements_cloudrun.txt

COPY src /app/src
COPY scripts /app/scripts
COPY streamlit_app /app/streamlit_app
COPY README.md /app/README.md

ENTRYPOINT ["python"]
CMD ["scripts/run_gcp_pipeline.py"]
