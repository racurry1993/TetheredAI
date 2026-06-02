# TetheredAI MLB GCP Fresh Start

This package resets the GCP deployment around one principle:

- GitHub stores code only.
- Google Cloud Storage stores large state: `odds.db`, features, predictions, model artifacts.
- Cloud Run Jobs execute the pipeline.
- Cloud Scheduler can trigger Cloud Run Jobs later.

The Cloud Run jobs are environment-variable driven. They do not rely on fragile `--args` lists.

## Files to copy into `MLB/mlb_betting_app/`

```text
Dockerfile
.dockerignore
.gcloudignore
requirements_cloudrun.txt
scripts/run_gcp_pipeline.py
scripts/gcs_download_state.py
scripts/gcs_upload_state.py
src/mlb_betting/gcs_state.py
gcp/cloudshell_start_fresh.sh
gcp/README_GCP_FRESH_START.md
```

Commit and push these files before using Cloud Shell:

```bash
git add MLB/mlb_betting_app/Dockerfile
git add MLB/mlb_betting_app/.dockerignore
git add MLB/mlb_betting_app/.gcloudignore
git add MLB/mlb_betting_app/requirements_cloudrun.txt
git add MLB/mlb_betting_app/scripts/run_gcp_pipeline.py
git add MLB/mlb_betting_app/scripts/gcs_download_state.py
git add MLB/mlb_betting_app/scripts/gcs_upload_state.py
git add MLB/mlb_betting_app/src/mlb_betting/gcs_state.py
git add MLB/mlb_betting_app/gcp/cloudshell_start_fresh.sh
git add MLB/mlb_betting_app/gcp/README_GCP_FRESH_START.md
git commit -m "Add fresh GCP Cloud Run pipeline"
git push
```

Make sure `.gitignore` contains:

```gitignore
data/odds.db
data/*.db
.env
```

## Cloud Shell reset steps

Open Cloud Shell, then start clean:

```bash
cd ~
rm -rf TetheredAI
git clone https://github.com/racurry1993/TetheredAI.git
cd TetheredAI/MLB/mlb_betting_app
```

Set project variables:

```bash
export PROJECT_ID="tetheredai-preds"
export REGION="us-central1"
export AR_REPO="tetheredai"
export BUCKET="tetheredai-mlb-state-${PROJECT_ID}"
export RUN_SA_NAME="tetheredai-mlb-runner"
export RUN_SA="${RUN_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/mlb-pipeline:latest"

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"
```

Run the setup script:

```bash
bash gcp/cloudshell_start_fresh.sh
```

The script will:

1. enable APIs
2. create the GCS bucket if needed
3. create the Artifact Registry repo if needed
4. create the service account if needed
5. create or reuse the Odds API secret
6. build and push the container image
7. delete old stale Cloud Run jobs
8. create three fresh jobs:
   - `tetheredai-mlb-smoke`
   - `tetheredai-mlb-feature-refresh`
   - `tetheredai-mlb-daily-score`

## Smoke test

Run:

```bash
gcloud run jobs execute tetheredai-mlb-smoke --region="$REGION" --wait
```

Check logs if needed:

```bash
EXECUTION="$(gcloud run jobs executions list --job=tetheredai-mlb-smoke --region="$REGION" --limit=1 --format='value(name)')"

gcloud logging read \
'resource.type="cloud_run_job"
resource.labels.job_name="tetheredai-mlb-smoke"
resource.labels.location="us-central1"
labels."run.googleapis.com/execution_name"="'$EXECUTION'"' \
--project="$PROJECT_ID" \
--limit=200 \
--order=asc \
--format='value(textPayload)'
```

## Tiny feature-refresh test

Run:

```bash
gcloud run jobs execute tetheredai-mlb-feature-refresh --region="$REGION" --wait
```

This runs only:

```text
DAYS_BACK=90
STATCAST_DAYS_BACK=14
STATCAST_LIMIT_CHUNKS=1
```

Check GCS:

```bash
gcloud storage ls "gs://${BUCKET}/mlb/**"
```

You want to see at least:

```text
gs://.../mlb/state/odds.db
gs://.../mlb/processed/mlb_game_features.parquet
```

## Update feature-refresh to full backfill

Only after the tiny test works:

```bash
gcloud run jobs update tetheredai-mlb-feature-refresh \
  --region="$REGION" \
  --update-env-vars="DAYS_BACK=730,STATCAST_DAYS_BACK=730,STATCAST_LIMIT_CHUNKS="
```

Then run:

```bash
gcloud run jobs execute tetheredai-mlb-feature-refresh --region="$REGION" --wait
```

If it fails after partial work, rerun the same command. The pipeline uploads `odds.db` immediately after the expensive Statcast step, and `SKIP_EXISTING_STATCAST=true` should avoid redoing loaded chunks if your `09_fetch_statcast.py` supports that option.

## Daily scoring job

Run manually:

```bash
gcloud run jobs execute tetheredai-mlb-daily-score --region="$REGION" --wait
```

If no champion model exists in GCS yet, scoring will skip safely.

Upload a champion model when ready:

```bash
gcloud storage cp models/mlb_moneyline_champion.joblib \
  "gs://${BUCKET}/mlb/models/mlb_moneyline_champion.joblib"

gcloud storage cp models/mlb_moneyline_champion_metadata.json \
  "gs://${BUCKET}/mlb/models/mlb_moneyline_champion_metadata.json"
```

## Logs for latest execution

Change `JOB_NAME` as needed:

```bash
export JOB_NAME="tetheredai-mlb-feature-refresh"
EXECUTION="$(gcloud run jobs executions list --job="$JOB_NAME" --region="$REGION" --limit=1 --format='value(name)')"
echo "$EXECUTION"

gcloud logging read \
'resource.type="cloud_run_job"
resource.labels.job_name="'$JOB_NAME'"
resource.labels.location="us-central1"
labels."run.googleapis.com/execution_name"="'$EXECUTION'"' \
--project="$PROJECT_ID" \
--limit=300 \
--order=asc \
--format='value(textPayload)'
```

## Scheduling later

After manual daily scoring works, grant invoker and create a scheduler job:

```bash
gcloud run jobs add-iam-policy-binding tetheredai-mlb-daily-score \
  --region="$REGION" \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/run.invoker"

gcloud scheduler jobs create http tetheredai-mlb-daily-score \
  --location="$REGION" \
  --schedule="35 8 * * *" \
  --time-zone="America/Chicago" \
  --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/tetheredai-mlb-daily-score:run" \
  --http-method=POST \
  --oauth-service-account-email="$RUN_SA"
```

Do not schedule the full feature refresh until it is stable manually.
