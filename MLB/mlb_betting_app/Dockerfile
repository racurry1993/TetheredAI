FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install google-cloud-storage>=2.16.0

COPY . .

ENTRYPOINT ["python"]
CMD ["scripts/run_gcp_pipeline.py", "--mode", "daily", "--download-state", "--upload-state", "--fetch-odds"]
