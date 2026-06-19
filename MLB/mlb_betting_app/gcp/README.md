# GCP MLB ingestion scripts

Upload this folder to your GitHub repo as `GCP/`.

Run from Cloud Shell after `git pull`:

```bash
cd ~/sports-edge
source .venv/bin/activate
pip install -r GCP/requirements.txt

export PROJECT_ID="tetheredai-preds"
export BUCKET="${PROJECT_ID}-sports-edge-data"
export BQ_LOCATION="US"
```

Smoke test:

```bash
python GCP/ingest_mlb_smoke_test.py \
  --bucket "$BUCKET" \
  --start-date 2023-03-30 \
  --end-date 2023-04-02 \
  --max-game-details 5

python GCP/load_mlb_to_bigquery.py \
  --project-id "$PROJECT_ID" \
  --bucket "$BUCKET" \
  --mode smoke \
  --start-date 2023-03-30 \
  --end-date 2023-04-02

python GCP/build_mlb_features.py \
  --project-id "$PROJECT_ID" \
  --mode smoke
```

Historical backfill after smoke test works:

```bash
python GCP/ingest_mlb_historical.py \
  --bucket "$BUCKET" \
  --start-date 2023-01-01 \
  --end-date "$(date +%F)" \
  --include-game-details \
  --include-statcast \
  --statcast-chunk-days 3 \
  --sleep-seconds 2 \
  --max-workers 6

python GCP/load_mlb_to_bigquery.py \
  --project-id "$PROJECT_ID" \
  --bucket "$BUCKET" \
  --mode historical

python GCP/build_mlb_features.py \
  --project-id "$PROJECT_ID" \
  --mode prod
```

Daily incremental:

```bash
python GCP/ingest_mlb_daily.py \
  --bucket "$BUCKET" \
  --lookback-days 3 \
  --include-game-details \
  --include-statcast

python GCP/load_mlb_to_bigquery.py \
  --project-id "$PROJECT_ID" \
  --bucket "$BUCKET" \
  --mode daily \
  --run-date "$(date +%F)"

python GCP/build_mlb_features.py \
  --project-id "$PROJECT_ID" \
  --mode prod
```
