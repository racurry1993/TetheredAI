# TetheredAI MLB on Google Cloud Storage + Cloud Run Jobs

This setup keeps your growing SQLite database out of GitHub.

- Code lives in GitHub.
- `data/odds.db`, predictions, feature parquet, and model artifacts live in Google Cloud Storage.
- Cloud Run Jobs execute the Python pipeline.
- Cloud Scheduler triggers jobs on a cron schedule.

## 0. Set variables

Run from Google Cloud Shell or any terminal with `gcloud` authenticated.

```bash
export PROJECT_ID="your-gcp-project-id"
export PROJECT_NUMBER="$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')"
export REGION="us-central1"
export BUCKET="tetheredai-mlb-state-${PROJECT_ID}"
export AR_REPO="tetheredai"
export RUN_SA="tetheredai-mlb-runner@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
```

## 1. Enable services

```bash
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com
```

## 2. Create bucket and Artifact Registry

```bash
gcloud storage buckets create "gs://${BUCKET}" \
  --location=US \
  --uniform-bucket-level-access

gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="TetheredAI containers"
```

## 3. Create service account and grant access

```bash
gcloud iam service-accounts create tetheredai-mlb-runner \
  --display-name="TetheredAI MLB Runner"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/logging.logWriter"
```

## 4. Store Odds API key in Secret Manager

```bash
printf "YOUR_ODDS_API_KEY" | gcloud secrets create odds-api-key --data-file=-

gcloud secrets add-iam-policy-binding odds-api-key \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

## 5. Copy files into your app

Copy these package files into `MLB/mlb_betting_app/`:

```text
src/mlb_betting/gcs_state.py
scripts/gcs_download_state.py
scripts/gcs_upload_state.py
scripts/run_gcp_pipeline.py
Dockerfile
.dockerignore
cloudbuild.yaml
```

Add this to `requirements.txt`:

```text
google-cloud-storage>=2.16.0
```

Make sure `.gitignore` includes:

```text
data/odds.db
data/*.db
.env
```

## 6. Build image

From `MLB/mlb_betting_app`:

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions _REGION=$REGION,_REPO=$AR_REPO,_IMAGE=mlb-pipeline
```

## 7. Create Cloud Run Jobs

### Daily score job

```bash
gcloud run jobs create tetheredai-mlb-daily-score \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/mlb-pipeline:latest" \
  --region="$REGION" \
  --service-account="$RUN_SA" \
  --cpu=2 \
  --memory=4Gi \
  --task-timeout=2h \
  --max-retries=0 \
  --set-env-vars="GCS_BUCKET=${BUCKET},GCS_USE_LOCK=true,ODDS_DB_PATH=data/odds.db,DATA_DIR=data,MODEL_DIR=models,ODDS_SPORT_KEY=baseball_mlb,ODDS_REGIONS=us,ODDS_FORMAT=american,FETCH_ODDS=true,DAYS_BACK=14,DAYS_FORWARD=3,REFRESH_STATCAST=false,SCORE_GAMES=true" \
  --set-secrets="ODDS_API_KEY=odds-api-key:latest" \
  --args="scripts/run_gcp_pipeline.py,--mode,daily,--download-state,--upload-state,--fetch-odds,--days-back,14,--days-forward,3"
```

### Full feature/statcast refresh job

```bash
gcloud run jobs create tetheredai-mlb-feature-refresh \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/mlb-pipeline:latest" \
  --region="$REGION" \
  --service-account="$RUN_SA" \
  --cpu=4 \
  --memory=8Gi \
  --task-timeout=8h \
  --max-retries=0 \
  --set-env-vars="GCS_BUCKET=${BUCKET},GCS_USE_LOCK=true,ODDS_DB_PATH=data/odds.db,DATA_DIR=data,MODEL_DIR=models,ODDS_SPORT_KEY=baseball_mlb,ODDS_REGIONS=us,ODDS_FORMAT=american,FETCH_ODDS=false,DAYS_BACK=730,DAYS_FORWARD=14,REFRESH_STATCAST=true,STATCAST_DAYS_BACK=730,STATCAST_CHUNK_DAYS=3,SKIP_EXISTING_STATCAST=true,SCORE_GAMES=false" \
  --set-secrets="ODDS_API_KEY=odds-api-key:latest" \
  --args="scripts/run_gcp_pipeline.py,--mode,feature-refresh,--download-state,--upload-state,--days-back,730,--days-forward,14,--refresh-statcast,--statcast-days-back,730,--statcast-chunk-days,3,--skip-existing-statcast"
```

## 8. Execute manually

```bash
gcloud run jobs execute tetheredai-mlb-daily-score --region="$REGION" --wait
```

For the first Statcast validation, use a shorter one by updating the feature refresh job:

```bash
gcloud run jobs update tetheredai-mlb-feature-refresh \
  --region="$REGION" \
  --update-env-vars="STATCAST_DAYS_BACK=14" \
  --args="scripts/run_gcp_pipeline.py,--mode,feature-refresh,--download-state,--upload-state,--days-back,90,--days-forward,14,--refresh-statcast,--statcast-days-back,14,--statcast-chunk-days,3,--statcast-limit-chunks,1,--skip-existing-statcast"

gcloud run jobs execute tetheredai-mlb-feature-refresh --region="$REGION" --wait
```

## 9. Schedule jobs

Daily scoring at 8:35 AM Central:

```bash
gcloud scheduler jobs create http tetheredai-mlb-daily-score \
  --location="$REGION" \
  --schedule="35 8 * * *" \
  --time-zone="America/Chicago" \
  --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/tetheredai-mlb-daily-score:run" \
  --http-method=POST \
  --oauth-service-account-email="$RUN_SA"
```

Weekly feature refresh Monday at 3:00 AM Central:

```bash
gcloud scheduler jobs create http tetheredai-mlb-feature-refresh \
  --location="$REGION" \
  --schedule="0 3 * * 1" \
  --time-zone="America/Chicago" \
  --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/tetheredai-mlb-feature-refresh:run" \
  --http-method=POST \
  --oauth-service-account-email="$RUN_SA"
```

## 10. GCS layout

The pipeline writes:

```text
gs://BUCKET/mlb/state/odds.db
gs://BUCKET/mlb/processed/mlb_game_features.parquet
gs://BUCKET/mlb/predictions/mlb_moneyline_predictions.csv
gs://BUCKET/mlb/models/mlb_moneyline_champion.joblib
gs://BUCKET/mlb/models/mlb_moneyline_champion_metadata.json
gs://BUCKET/mlb/artifacts/features/<run_id>/mlb_game_features.parquet
gs://BUCKET/mlb/artifacts/predictions/<run_id>/mlb_moneyline_predictions.csv
```

## Notes

- Do not commit `data/odds.db` to GitHub.
- Do not run the daily and feature refresh jobs at the same time; the GCS lock helps prevent this.
- `odds.db` in GCS is a transitional architecture. Long-term, Postgres or BigQuery is cleaner.
